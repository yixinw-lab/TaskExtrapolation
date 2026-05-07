import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TASKS_EVAL = 500  

# --- Helper ---
def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)

# ==========================================
# 1. Library & Task2Vec
# ==========================================

class UniversalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([1, 64, 64, 1])
    def forward(self, x): return self.net(x)

class EmbeddingEngine:
    def __init__(self, library, method="task2vec", z_dim=16, seed=42):
        self.method = method
        self.z_dim = z_dim
        self.input_domain = torch.tensor(library.x_range).float().reshape(-1, 1).to(DEVICE)
        
        print(f"--- Initializing Embedding Engine: {method} ---")
        
        if method == "random":
            self.embeddings = np.random.randn(len(library.dense_curves), z_dim)
            self._normalize()
        elif method == "raw_pca":
            pca = PCA(n_components=z_dim)
            self.embeddings = pca.fit_transform(library.dense_curves)
            self._normalize()
        elif method == "task2vec":
            self._compute_task2vec(library)
            self._normalize()

        self.nn_engine = NearestNeighbors(metric='euclidean')
        self.nn_engine.fit(self.embeddings)

    def _normalize(self):
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = (self.embeddings / (norms + 1e-8)).astype(np.float32)

    def _compute_task2vec(self, library):
        probe = UniversalProbe().to(DEVICE)
        opt = optim.Adam(probe.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        print("Pre-training Probe...")
        for _ in range(500):
            idx = np.random.randint(len(library.dense_curves))
            y = torch.from_numpy(library.dense_curves[idx]).float().reshape(-1, 1).to(DEVICE)
            loss_fn(probe(self.input_domain), y).backward()
            opt.step(); opt.zero_grad()
            
        print("Extracting Gradients...")
        grads = []
        for i in range(len(library.dense_curves)):
            probe.zero_grad()
            y = torch.from_numpy(library.dense_curves[i]).float().reshape(-1, 1).to(DEVICE)
            loss_fn(probe(self.input_domain), y).backward()
            g = []
            for p in probe.parameters():
                if p.grad is not None: g.append(p.grad.view(-1).cpu().numpy())
            grads.append(np.concatenate(g))
            
        pca = PCA(n_components=self.z_dim)
        self.embeddings = pca.fit_transform(np.stack(grads))

class PrimitiveLibrary:
    def __init__(self, n_per_family=1000, L=50, seed=42): 
        self.rng = np.random.default_rng(seed)
        self.L = L
        self.x_range = np.linspace(-5, 5, L).astype(np.float32)
        self.functions = []  
        self.dense_curves = [] 
        
        # Poly
        for _ in range(n_per_family):
            a, b, c = self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.5, 0.5), self.rng.uniform(-2, 2)
            f = lambda x, a=a, b=b, c=c: a*x**2 + b*x + c
            self.functions.append({'func': f, 'type': 'poly'})
            self.dense_curves.append(f(self.x_range))
        # Sin
        for _ in range(n_per_family):
            A, w, phi = self.rng.uniform(0.5, 2.0), self.rng.uniform(0.5, 1.5), self.rng.uniform(0, np.pi)
            f = lambda x, A=A, w=w, phi=phi: A * np.sin(w*x + phi)
            self.functions.append({'func': f, 'type': 'sin'})
            self.dense_curves.append(f(self.x_range))
        # Tanh
        for _ in range(n_per_family):
            A, w, c = self.rng.uniform(1.0, 3.0), self.rng.uniform(0.5, 2.0), self.rng.uniform(-2, 2)
            f = lambda x, A=A, w=w, c=c: A * np.tanh(w*(x - c))
            self.functions.append({'func': f, 'type': 'tanh'})
            self.dense_curves.append(f(self.x_range))

        self.dense_curves = np.stack(self.dense_curves)

# ==========================================
# 2. Models
# ==========================================

class CompDecomposer(nn.Module):
    def __init__(self, L=50, z_dim=16):
        super().__init__()
        in_dim = (L // 2) * 2 
        self.shared = mlp([in_dim, 256, 128, 64])
        self.head_f = nn.Linear(64, z_dim)
        self.head_g = nn.Linear(64, z_dim)
        
    def forward(self, x, h):
        sort_idx = torch.argsort(x, dim=1)
        fin = torch.cat([torch.gather(x, 1, sort_idx), torch.gather(h, 1, sort_idx)], dim=-1)
        feat = self.shared(fin)
        return self.head_f(feat), self.head_g(feat)

class CompOperator(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        # Inputs: x(L) + y_f_desc(L) + y_g_desc(L) + y_head(L/2)
        input_dim = L + L + L + (L // 2)
        self.block1 = mlp([input_dim, 256, 256, 128]) 
        self.block2 = mlp([128, 128, 64])
        self.out = mlp([64, 64, L // 2]) 

    def forward(self, x_query, y_f_desc, y_g_desc, y_target_head):
        fin = torch.cat([x_query, y_f_desc, y_g_desc, y_target_head], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

class NaiveBaseline(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        input_dim = L + (L // 2)
        self.block1 = mlp([input_dim, 256, 256, 128])
        self.block2 = mlp([128, 128, 64])
        self.out = mlp([64, 64, L // 2])

    def forward(self, x, h_head):
        fin = torch.cat([x, h_head], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

# ==========================================
# 3. Training Routines
# ==========================================

def generate_batch(library, batch_size, emb_engine):
    L = library.L
    inds = np.arange(len(library.functions))
    
    x_query = library.rng.uniform(-5.0, 5.0, size=(batch_size, L)).astype(np.float32)
    
    batch_X, batch_Yf, batch_Yg = [], [], []
    batch_Hh, batch_Ht = [], []
    batch_Zf, batch_Zg = [], []
    
    for i in range(batch_size):
        idx_f = np.random.choice(inds)
        idx_g = np.random.choice(inds)
        
        f_obj = library.functions[idx_f]
        g_obj = library.functions[idx_g]
        
        y_h_query = f_obj['func'](g_obj['func'](x_query[i])).astype(np.float32)
        
        batch_X.append(x_query[i])
        batch_Hh.append(y_h_query[:L//2])
        batch_Ht.append(y_h_query[L//2:])
        batch_Yf.append(library.dense_curves[idx_f])
        batch_Yg.append(library.dense_curves[idx_g])
        batch_Zf.append(emb_engine.embeddings[idx_f])
        batch_Zg.append(emb_engine.embeddings[idx_g])
        
    return (
        torch.tensor(np.stack(batch_X)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Yf)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Yg)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Hh)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Ht)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Zf)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Zg)).float().to(DEVICE),
    )

def train_baseline(lib, steps=2000):
    model = NaiveBaseline(lib.L).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    for _ in range(steps):
        x = np.random.uniform(-5, 5, (64, lib.L)).astype(np.float32)
        h = []
        for i in range(64):
            f = lib.functions[np.random.randint(len(lib.functions))]['func']
            g = lib.functions[np.random.randint(len(lib.functions))]['func']
            h.append(f(g(x[i])))
        H = torch.tensor(np.stack(h)).float().to(DEVICE)
        X = torch.tensor(x).float().to(DEVICE)
        loss = nn.MSELoss()(model(X, H[:, :lib.L//2]), H[:, lib.L//2:])
        loss.backward(); opt.step(); opt.zero_grad()
    return model

def train_transductive(lib, emb_engine, steps=3000):
    decomp = CompDecomposer(lib.L, emb_engine.z_dim).to(DEVICE)
    op = CompOperator(lib.L).to(DEVICE)
    opt = optim.Adam(list(decomp.parameters()) + list(op.parameters()), lr=1e-3)
    
    for _ in range(steps):
        X, Yf, Yg, Hh, Ht, Zf, Zg = generate_batch(lib, 64, emb_engine)
        X_head = X[:, :lib.L//2]
        
        # Train Decomposer
        zf_p, zg_p = decomp(X_head, Hh)
        loss_z = nn.MSELoss()(zf_p, Zf) + nn.MSELoss()(zg_p, Zg)
        
        # Train Operator
        pred = op(X, Yf, Yg, Hh)
        loss_y = nn.MSELoss()(pred, Ht)
        
        (loss_z + loss_y).backward()
        opt.step(); opt.zero_grad()
        
    return decomp, op

# ==========================================
# 4. Evaluation Suite
# ==========================================

class ExperimentSuite:
    def __init__(self):
        self.lib = PrimitiveLibrary(n_per_family=1000)
        self.results = []
        
    def run_eval(self, model, emb_engine, name, rank_n=1, no_verify=False, oracle=False):
        is_baseline = isinstance(model, NaiveBaseline)
        if not is_baseline:
            decomp, op = model
            decomp.eval(); op.eval()
        else:
            model.eval()
            
        errors = []
        
        for _ in range(N_TASKS_EVAL):
            # 1. Create OOD Task
            idx_f_true = np.random.randint(len(self.lib.functions))
            idx_g_true = np.random.randint(len(self.lib.functions))
            f_true = self.lib.functions[idx_f_true]['func']
            g_true = self.lib.functions[idx_g_true]['func']
            
            x_query = np.random.uniform(-5, 5, (1, self.lib.L)).astype(np.float32)
            h_true_curve = f_true(g_true(x_query))
            h_head = h_true_curve[:, :self.lib.L//2]
            h_tail = h_true_curve[:, self.lib.L//2:]
            
            X_t = torch.tensor(x_query).float().to(DEVICE)
            H_head_t = torch.tensor(h_head).float().to(DEVICE)
            X_head_t = X_t[:, :self.lib.L//2]
            
            if is_baseline:
                with torch.no_grad():
                    pred = model(X_t, H_head_t).cpu().numpy()
            else:
                with torch.no_grad():
                    if oracle:
                        # ORACLE: Use Ground Truth indices
                        best_f = idx_f_true
                        best_g = idx_g_true
                    else:
                        # STANDARD: Infer and Search
                        zf_p, zg_p = decomp(X_head_t, H_head_t)
                        
                        if rank_n == 'random':
                            best_f = np.random.randint(len(self.lib.functions))
                            best_g = np.random.randint(len(self.lib.functions))
                        else:
                            # B. Retrieval (Top K)
                            k_search = 10
                            _, inds_f = emb_engine.nn_engine.kneighbors(zf_p.cpu().numpy(), n_neighbors=k_search)
                            _, inds_g = emb_engine.nn_engine.kneighbors(zg_p.cpu().numpy(), n_neighbors=k_search)
                            cand_f = inds_f[0]
                            cand_g = inds_g[0]
                            
                            if no_verify:
                                best_f = cand_f[rank_n - 1]
                                best_g = cand_g[rank_n - 1]
                            else:
                                # C. PHYSICS VERIFICATION
                                scores = []
                                for i in range(len(cand_f)):
                                    for j in range(len(cand_g)):
                                        cf, cg = cand_f[i], cand_g[j]
                                        func_f = self.lib.functions[cf]['func']
                                        func_g = self.lib.functions[cg]['func']
                                        
                                        x_head_np = x_query[0, :self.lib.L//2]
                                        h_cand = func_f(func_g(x_head_np))
                                        
                                        mse_ctx = np.mean((h_cand - h_head[0])**2)
                                        scores.append((mse_ctx, cf, cg))
                                
                                scores.sort(key=lambda x: x[0])
                                picked = scores[min(rank_n - 1, len(scores)-1)]
                                best_f, best_g = picked[1], picked[2]

                    # D. Operator Prediction
                    Yf_in = torch.tensor(self.lib.dense_curves[best_f]).unsqueeze(0).float().to(DEVICE)
                    Yg_in = torch.tensor(self.lib.dense_curves[best_g]).unsqueeze(0).float().to(DEVICE)
                    
                    pred = op(X_t, Yf_in, Yg_in, H_head_t).cpu().numpy()

            mse = np.mean((pred - h_tail)**2)
            errors.append(mse)
            
        mean = np.mean(errors)
        ci = 1.96 * np.std(errors) / np.sqrt(len(errors))
        print(f"[{name}] MSE: {mean:.4f} ± {ci:.4f}")
        self.results.append({"Method": name, "MSE": mean, "CI": ci})

    def run_all(self):
        print(">>> Training Baseline...")
        base = train_baseline(self.lib)
        self.run_eval(base, None, "Inductive Baseline")
        
        print("\n>>> Training Transductive (Task2Vec)...")
        emb = EmbeddingEngine(self.lib, "task2vec") 
        trans = train_transductive(self.lib, emb)
        
        self.run_eval(trans, emb, "Transductive (Standard)", rank_n=1)
        self.run_eval(trans, emb, "Ablation: Rank-3 Pair", rank_n=3)
        self.run_eval(trans, emb, "Ablation: Random Pair", rank_n='random')
        self.run_eval(trans, emb, "Ablation: No Verification", rank_n=1, no_verify=True)
        # Added Oracle
        self.run_eval(trans, emb, "Upper Bound: Oracle", oracle=True)
        
        print("\n>>> Training Transductive (Raw PCA)...")
        emb_pca = EmbeddingEngine(self.lib, "raw_pca")
        trans_pca = train_transductive(self.lib, emb_pca)
        self.run_eval(trans_pca, emb_pca, "Ablation: Raw PCA Embedding")
        
        pd.DataFrame(self.results).to_csv("comp_ablation_final.csv", index=False)

if __name__ == "__main__":
    ExperimentSuite().run_all()
