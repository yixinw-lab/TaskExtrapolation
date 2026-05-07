import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TASKS_EVAL = 500  

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
# 1. Library & Embeddings
# ==========================================

class UniversalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([1, 64, 64, 1])
    def forward(self, x): return self.net(x)

class Task2VecEncoder:
    def __init__(self, input_domain, z_dim=16, seed=42):
        self.z_dim = z_dim
        self.input_domain = torch.tensor(input_domain).float().reshape(-1, 1).to(DEVICE)
        torch.manual_seed(seed)
        self.probe = UniversalProbe().to(DEVICE)
        self.pca = PCA(n_components=z_dim)

    def fit_transform_library(self, library):
        # 1. Pretrain Probe
        self.probe.train()
        opt = optim.Adam(self.probe.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        n_funcs = len(library.dense_curves)
        print("Pre-training Task2Vec Probe...")
        for _ in range(500): 
            idx = np.random.randint(n_funcs)
            y_true = torch.from_numpy(library.dense_curves[idx]).float().reshape(-1, 1).to(DEVICE)
            opt.zero_grad()
            loss_fn(self.probe(self.input_domain), y_true).backward()
            opt.step()
            
        # 2. Extract Gradients
        print("Extracting Gradients...")
        grads = []
        for i in range(n_funcs):
            self.probe.zero_grad()
            y_true = torch.from_numpy(library.dense_curves[i]).float().reshape(-1, 1).to(DEVICE)
            loss_fn(self.probe(self.input_domain), y_true).backward()
            g = []
            for p in self.probe.parameters():
                if p.grad is not None: g.append(p.grad.view(-1).cpu().numpy())
            grads.append(np.concatenate(g))
            
        # 3. PCA
        self.embeddings = self.pca.fit_transform(np.stack(grads))
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        
        self.nn_engine = NearestNeighbors(metric='euclidean')
        self.nn_engine.fit(self.embeddings)
        return self.embeddings

class PolyLibrary:
    def __init__(self, n_per_degree=500, max_degree=8, L=16, seed=42):
        self.rng = np.random.default_rng(seed)
        self.L = L
        self.x_range = np.linspace(-5, 5, L).astype(np.float32)
        self.functions = []  
        self.dense_curves = []
        
        print(f"Populating Library (Degrees 1-{max_degree})...")
        for d in range(1, max_degree + 1):
            self._populate_degree(d, n_per_degree)
        self.dense_curves = np.stack(self.dense_curves)
        
    def _sample_coeffs(self, degree):
        c_list = []
        for p in range(degree + 1):
            scale = 1.0 / (5.0 ** p) if p > 0 else 1.0
            val = self.rng.uniform(-2.0, 2.0) * scale
            c_list.append(val)
        return np.array(c_list)

    def _populate_degree(self, degree, n):
        for _ in range(n):
            c = self._sample_coeffs(degree)
            def f(x, c=c):
                y = np.zeros_like(x)
                for p, val in enumerate(c): y += val * (x ** p)
                return y
            self.functions.append({'func': f, 'degree': degree, 'coeffs': c})
            self.dense_curves.append(f(self.x_range))

# ==========================================
# 2. Models
# ==========================================

class LengthDecomposer(nn.Module):
    def __init__(self, L=16, z_dim=16):
        super().__init__()
        in_dim = (L // 2) * 2 
        self.shared = mlp([in_dim, 256, 128, 64])
        self.head_z = nn.Linear(64, z_dim)
        self.head_c = nn.Linear(64, 1)
    def forward(self, x, h):
        sort_idx = torch.argsort(x, dim=1)
        fin = torch.cat([torch.gather(x, 1, sort_idx), torch.gather(h, 1, sort_idx)], dim=-1)
        feat = self.shared(fin)
        return self.head_z(feat), self.head_c(feat)

class LengthExtender(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        self.net = mlp([L + L + 1, 256, 256, 128, 128, 128, 64, 64, L])
    def forward(self, x, y_prev, c_new):
        return self.net(torch.cat([x, y_prev, c_new], dim=-1))

class NaiveBaseline(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        self.net = mlp([L + (L // 2), 256, 256, 128, 128, 128, 64, 64, L // 2])
    def forward(self, x, h_head):
        return self.net(torch.cat([x, h_head], dim=-1)).squeeze(-1)

# ==========================================
# 3. Training & Eval
# ==========================================

def train_system(lib, encoder, steps=3000):
    decomp = LengthDecomposer(L=lib.L).to(DEVICE)
    ext = LengthExtender(L=lib.L).to(DEVICE)
    base = NaiveBaseline(L=lib.L).to(DEVICE)
    
    opt_sys = optim.Adam(list(decomp.parameters()) + list(ext.parameters()), lr=1e-3)
    opt_base = optim.Adam(base.parameters(), lr=1e-3)
    
    for _ in range(steps):
        idxs = np.random.choice(len(lib.functions), 64)
        funcs = [lib.functions[i] for i in idxs]
        
        x_q = np.random.uniform(-5, 5, (64, lib.L)).astype(np.float32)
        y_prev, y_tgt, z_prev, c_true = [], [], [], []
        
        for i, f in enumerate(funcs):
            d = f['degree']
            s = 1.0 / (5.0**(d+1))
            c = np.random.uniform(-2,2) * s
            yp = f['func'](x_q[i])
            yt = yp + c * (x_q[i]**(d+1))
            
            y_prev.append(yp); y_tgt.append(yt)
            z_prev.append(encoder.embeddings[idxs[i]])
            c_true.append(c * (5.0**(d+1)))
            
        X = torch.tensor(x_q).float().to(DEVICE)
        Yp = torch.tensor(np.stack(y_prev)).float().to(DEVICE)
        Yt = torch.tensor(np.stack(y_tgt)).float().to(DEVICE)
        Z = torch.tensor(np.stack(z_prev)).float().to(DEVICE)
        C = torch.tensor(np.stack(c_true)).float().to(DEVICE).unsqueeze(1)
        H_head = Yt[:, :lib.L//2]
        
        # Train Transductive
        zp, cp = decomp(X[:, :lib.L//2], H_head)
        loss_d = nn.MSELoss()(zp, Z) + nn.MSELoss()(cp, C)
        loss_e = nn.MSELoss()(ext(X, Yp, C), Yt)
        opt_sys.zero_grad(); (loss_d + loss_e).backward(); opt_sys.step()
        
        # Train Baseline
        opt_base.zero_grad()
        loss_b = nn.MSELoss()(base(X, H_head), Yt[:, lib.L//2:])
        loss_b.backward(); opt_base.step()
        
    return decomp, ext, base

class ExperimentSuite:
    def __init__(self):
        self.lib = PolyLibrary()
        self.t2v = Task2VecEncoder(self.lib.x_range)
        self.t2v.fit_transform_library(self.lib)
        self.results = []
        
    def run_eval(self, model, name, rank_n=1, no_verify=False, random_anchor=False, zero_shift=False, oracle=False):
        if name == "Inductive Baseline":
            model.eval()
        else:
            decomp, ext = model
            decomp.eval(); ext.eval()
            
        errors = []
        for _ in range(N_TASKS_EVAL):
            # Create Degree 9 Task
            d8_idxs = [i for i, f in enumerate(self.lib.functions) if f['degree'] == 8]
            f_prev_true = self.lib.functions[np.random.choice(d8_idxs)]
            c_val = np.random.uniform(-2, 2) * (1.0/5**9)
            
            x_q = np.random.uniform(-5, 5, (1, 16)).astype(np.float32)
            yp = f_prev_true['func'](x_q) # (1, 16)
            yt = yp + c_val * (x_q**9)
            
            X = torch.tensor(x_q).float().to(DEVICE)
            Hh = torch.tensor(yt[:, :8]).float().to(DEVICE)
            
            if name == "Inductive Baseline":
                with torch.no_grad(): pred = model(X, Hh).cpu().numpy().flatten()
            else:
                with torch.no_grad():
                    if oracle:
                        # ORACLE: Perfect anchor, Perfect Coeff
                        # yp is already (1, 16), so we don't need unsqueeze(0)
                        yp_t = torch.tensor(yp).float().to(DEVICE)
                        
                        c_norm = c_val * (5.0**9)
                        c_t = torch.tensor([[c_norm]]).float().to(DEVICE)
                        
                        pred = ext(X, yp_t, c_t)[0, 8:].cpu().numpy()
                    else:
                        # Standard inference
                        zp, cp = decomp(X[:, :8], Hh)
                        
                        if random_anchor:
                            best_idx = np.random.randint(len(self.lib.functions))
                        else:
                            # 1. Retrieval
                            k = 20
                            _, inds = self.t2v.nn_engine.kneighbors(zp.cpu().numpy(), n_neighbors=k)
                            candidates = inds[0]
                            
                            if no_verify:
                                best_idx = candidates[rank_n-1]
                            else:
                                # 2. Physics Verification (Neural)
                                scores = []
                                for idx in candidates:
                                    cand_f = self.lib.functions[idx]['func']
                                    y_cand = cand_f(x_q.flatten())
                                    y_cand_t = torch.tensor(y_cand).float().to(DEVICE).unsqueeze(0)
                                    
                                    # Use Extender to predict HEAD
                                    full_p = ext(X, y_cand_t, cp)
                                    head_p = full_p[:, :8]
                                    mse = nn.MSELoss()(head_p, Hh).item()
                                    scores.append((mse, idx))
                                
                                scores.sort(key=lambda x: x[0])
                                best_idx = scores[min(rank_n-1, len(scores)-1)][1]
                        
                        # 3. Final Prediction
                        y_best = self.lib.functions[best_idx]['func'](x_q.flatten())
                        y_best_t = torch.tensor(y_best).float().to(DEVICE).unsqueeze(0)
                        
                        if zero_shift:
                            c_zero = torch.zeros_like(cp)
                            pred = ext(X, y_best_t, c_zero)[0, 8:].cpu().numpy()
                        else:
                            pred = ext(X, y_best_t, cp)[0, 8:].cpu().numpy()
                        
            gt = yt[0, 8:]
            errors.append(np.mean((pred - gt)**2))
            
        mean = np.mean(errors)
        ci = 1.96 * np.std(errors)/np.sqrt(len(errors))
        print(f"[{name}] MSE: {mean:.4f} ± {ci:.4f}")
        self.results.append({"Method": name, "MSE": mean, "CI": ci})

    def run_all(self):
        print(">>> Training Systems...")
        decomp, ext, base = train_system(self.lib, self.t2v)
        
        self.run_eval(base, "Inductive Baseline")
        self.run_eval((decomp, ext), "Transductive (Standard)", rank_n=1)
        self.run_eval((decomp, ext), "Ablation: Rank-3 Anchor", rank_n=3)
        self.run_eval((decomp, ext), "Ablation: Random Anchor", random_anchor=True)
        self.run_eval((decomp, ext), "Ablation: Zero Shift (Neural)", rank_n=1, zero_shift=True)
        self.run_eval((decomp, ext), "Upper Bound: Oracle", oracle=True)
        
        pd.DataFrame(self.results).to_csv("length_ablation_final.csv", index=False)

if __name__ == "__main__":
    ExperimentSuite().run_all()
