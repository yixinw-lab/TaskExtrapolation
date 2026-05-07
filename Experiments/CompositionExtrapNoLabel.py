import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Neural Network Helpers ---

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
# Task2Vec with PRE-TRAINING
# ==========================================

class UniversalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([1, 64, 64, 1])
        
    def forward(self, x):
        return self.net(x)

class Task2VecEncoder:
    def __init__(self, input_domain, z_dim=16, seed=42):
        self.z_dim = z_dim
        self.input_domain = torch.tensor(input_domain).float().reshape(-1, 1).to(DEVICE)
        torch.manual_seed(seed)
        self.probe = UniversalProbe().to(DEVICE)
        self.pca = PCA(n_components=z_dim)

    def pretrain_probe(self, library, iterations=1000, lr=1e-3):
        print(f"Pre-training Task2Vec probe for {iterations} steps...")
        self.probe.train()
        opt = optim.Adam(self.probe.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n_funcs = len(library.functions)
        for _ in range(iterations):
            idx = np.random.randint(n_funcs)
            y_true = torch.from_numpy(library.dense_curves[idx]).float().reshape(-1, 1).to(DEVICE)
            opt.zero_grad()
            y_pred = self.probe(self.input_domain)
            loss_fn(y_pred, y_true).backward()
            opt.step()

    def _get_raw_gradients(self, func_idx, library):
        self.probe.zero_grad()
        y_true = torch.from_numpy(library.dense_curves[func_idx]).float().reshape(-1, 1).to(DEVICE)
        y_pred = self.probe(self.input_domain)
        nn.MSELoss()(y_pred, y_true).backward()
        grads = []
        for p in self.probe.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1).cpu().numpy())
            else:
                grads.append(np.zeros(p.numel()))
        return np.concatenate(grads)

    def fit_transform_library(self, library):
        self.pretrain_probe(library)
        print("Computing gradient embeddings...")
        raw_grads = [self._get_raw_gradients(i, library) for i in range(len(library.functions))]
        raw_grads = np.stack(raw_grads)
        print("Fitting PCA...")
        self.embeddings = self.pca.fit_transform(raw_grads)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        return self.embeddings

# --- 1. The Primitive Library ---

class PrimitiveLibrary:
    def __init__(self, n_per_family=1000, L=50, seed=42): 
        self.rng = np.random.default_rng(seed)
        self.L = L
        self.x_range = np.linspace(-5, 5, L).astype(np.float32)
        self.functions = []  
        self.dense_curves = [] 
        
        print(f"Populating Library ({n_per_family}/family)...")
        self._populate(n_per_family)
        self.dense_curves = np.stack(self.dense_curves) 
        self.types_list = [f['type'] for f in self.functions]

    def _populate(self, n):
        # Poly
        for _ in range(n):
            a, b, c = self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.5, 0.5), self.rng.uniform(-2, 2)
            f = lambda x, a=a, b=b, c=c: a*x**2 + b*x + c
            self.functions.append({'func': f, 'type': 'poly'})
            self.dense_curves.append(f(self.x_range))
        # Sin
        for _ in range(n):
            A, w, phi = self.rng.uniform(0.5, 2.0), self.rng.uniform(0.5, 1.5), self.rng.uniform(0, np.pi)
            f = lambda x, A=A, w=w, phi=phi: A * np.sin(w*x + phi)
            self.functions.append({'func': f, 'type': 'sin'})
            self.dense_curves.append(f(self.x_range))
        # Tanh
        for _ in range(n):
            A, w, c = self.rng.uniform(1.0, 3.0), self.rng.uniform(0.5, 2.0), self.rng.uniform(-2, 2)
            f = lambda x, A=A, w=w, c=c: A * np.tanh(w*(x - c))
            self.functions.append({'func': f, 'type': 'tanh'})
            self.dense_curves.append(f(self.x_range))

    def compute_task2vec_embeddings(self):
        self.encoder = Task2VecEncoder(self.x_range, z_dim=16)
        self.embeddings = self.encoder.fit_transform_library(self)
        self.nn_engine = NearestNeighbors(metric='euclidean')
        self.nn_engine.fit(self.embeddings)

    def retrieve_top_k(self, z_query, k=10):
        _, indices = self.nn_engine.kneighbors(z_query.reshape(1, -1), n_neighbors=k)
        indices = indices[0]
        curves = self.dense_curves[indices]
        funcs = [self.functions[i]['func'] for i in indices]
        return curves, funcs

# --- 2. Models ---

class Decomposer(nn.Module):
    def __init__(self, L=50, z_dim=16):
        super().__init__()
        # Input: sorted x(L/2) + sorted h(L/2)
        n_obs = L // 2
        in_dim = n_obs * 2 
        self.shared = mlp([in_dim, 256, 128, 64])
        self.head_outer = nn.Linear(64, z_dim)
        self.head_inner = nn.Linear(64, z_dim)
        
    def forward(self, x, h):
        sort_idx = torch.argsort(x, dim=1)
        x_sorted = torch.gather(x, 1, sort_idx)
        h_sorted = torch.gather(h, 1, sort_idx)
        fin = torch.cat([x_sorted, h_sorted], dim=-1)
        feat = self.shared(fin)
        return self.head_outer(feat), self.head_inner(feat)

class CompositionModel(nn.Module):
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

class InductiveBaseline(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        input_dim = L + (L // 2)
        self.block1 = mlp([input_dim, 256, 256, 128])
        self.block2 = mlp([128, 128, 64])
        self.out = mlp([64, 64, L // 2])

    def forward(self, x, h_head):
        fin = torch.cat([x, h_head], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

# --- 3. Data Generation ---

def generate_batch_contrastive(library, batch_size):
    # Note: We keep the structure but we essentially don't need negatives anymore
    # since we removed the siamese/triplet loss.
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
        batch_Zf.append(library.embeddings[idx_f])
        batch_Zg.append(library.embeddings[idx_g])
        
    return (
        torch.tensor(np.stack(batch_X)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Yf)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Yg)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Hh)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Ht)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Zf)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Zg)).float().to(DEVICE)
    )

# --- 4. Logic & Re-Ranking ---

def select_best_pair(library, z_f, z_g, h_head_true, x_head_query, k=50):
    curves_f, funcs_f = library.retrieve_top_k(z_f, k=k)
    curves_g, funcs_g = library.retrieve_top_k(z_g, k=k)
    
    best_error = float('inf')
    best_f, best_g = curves_f[0], curves_g[0]
    target = h_head_true.cpu().numpy().flatten()
    
    for i in range(k):
        for j in range(k):
            h_candidate = funcs_f[i](funcs_g[j](x_head_query))
            mse = np.mean((h_candidate - target)**2)
            if mse < best_error:
                best_error = mse
                best_f, best_g = curves_f[i], curves_g[j]
    return best_f, best_g

def compare_models(library, decomp_rte, composer, inductive_model, n_trials=1000):
    print(f"\n--- Comparing Inductive vs RTE ({n_trials} trials) ---")
    mse_rte, mse_ind = [], []
    
    decomp_rte.eval()
    composer.eval()
    inductive_model.eval()
    
    L = library.L
    
    with torch.no_grad():
        for i in range(n_trials):
            # 1. Generate Task
            out = generate_batch_contrastive(library, batch_size=1)
            X, _, _, H_head, H_tail, _, _ = out
            X_head = X[:, :L//2]
            
            # 2. Inductive Baseline
            pred_ind = inductive_model(X, H_head)
            loss_ind = nn.MSELoss()(pred_ind, H_tail).item()
            mse_ind.append(loss_ind)
            
            # 3. RTE (Transductive)
            zf_pred, zg_pred = decomp_rte(X_head, H_head)
            bf, bg = select_best_pair(library, zf_pred.cpu().numpy()[0], zg_pred.cpu().numpy()[0], 
                                      H_head[0], X_head.cpu().numpy()[0], k=50)
            
            in_yf = torch.tensor(bf).float().reshape(1, -1).to(DEVICE)
            in_yg = torch.tensor(bg).float().reshape(1, -1).to(DEVICE)
            pred_rte = composer(X, in_yf, in_yg, H_head)
            mse_rte.append(nn.MSELoss()(pred_rte, H_tail).item())

    print("-" * 60)
    print(f"{'Model':<20} | {'Mean MSE':<12} | {'Std Dev':<12}")
    print("-" * 60)
    print(f"{'Inductive (Base)':<20} | {np.mean(mse_ind):.4f}        | {np.std(mse_ind):.4f}")
    print(f"{'RTE (Transductive)':<20} | {np.mean(mse_rte):.4f}        | {np.std(mse_rte):.4f}")
    print("-" * 60)

def visualize_comparison(outdir, library, decomp_rte, composer, inductive_model):
    # --- Config for Larger and Bolder Fonts ---
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'figure.figsize': (24, 8) 
    })

    decomp_rte.eval()
    composer.eval()
    inductive_model.eval()
    
    # --- Generate 10 Images ---
    for eval_idx in range(1, 51):
        idx_f, idx_g = np.random.randint(len(library.functions)), np.random.randint(len(library.functions))
        f_obj, g_obj = library.functions[idx_f], library.functions[idx_g]
        
        L = library.L
        X_test_np = library.rng.uniform(-5.0, 5.0, size=(200, L)).astype(np.float32)
        y_h_test = f_obj['func'](g_obj['func'](X_test_np)).astype(np.float32)
        
        X_test = torch.tensor(X_test_np).float().to(DEVICE)
        H_head = torch.tensor(y_h_test[:, :L//2]).float().to(DEVICE)
        H_tail = y_h_test[:, L//2:]
        X_head_only = X_test[:, :L//2]

        # Inductive Prediction
        ind_flat = inductive_model(X_test, H_head).detach().cpu().numpy().flatten()
        
        # RTE Retrieval
        zf_pred, zg_pred = decomp_rte(X_head_only, H_head)
        zf_pred, zg_pred = zf_pred.detach().cpu().numpy(), zg_pred.detach().cpu().numpy()
        
        rte_preds = []
        best_f_vis, best_g_vis = None, None

        # Pixel-wise Loop for RTE
        for i in range(200):
            x_hq = X_test_np[i, :L//2]
            
            # Select best components
            bf, bg = select_best_pair(library, zf_pred[i], zg_pred[i], H_head[i], x_hq, k=50)
            
            # Compose
            p_rte = composer(X_test[i:i+1], 
                             torch.tensor(bf).float().to(DEVICE).unsqueeze(0), 
                             torch.tensor(bg).float().to(DEVICE).unsqueeze(0), 
                             H_head[i:i+1])
            rte_preds.append(p_rte)
            
            if i == 0: best_f_vis, best_g_vis = bf, bg

        rte_flat = torch.cat(rte_preds).detach().cpu().numpy().flatten()
        true_flat = H_tail.flatten()
        
        mse_ind = np.mean((ind_flat - true_flat)**2)
        mse_rte = np.mean((rte_flat - true_flat)**2)

        # PLOTTING - 3 Graphs per Image
        fig, axes = plt.subplots(1, 3)
        
        # Graph 1: Anchor Comparison (True vs RTE Selected)
        ax = axes[0]
        ax.plot(library.x_range, library.dense_curves[idx_f], 'b-', lw=3, label='True f')
        ax.plot(library.x_range, best_f_vis, 'b--', lw=2, label='RTE f')
        ax.plot(library.x_range, library.dense_curves[idx_g], 'g-', lw=3, label='True g')
        ax.plot(library.x_range, best_g_vis, 'g--', lw=2, label='RTE g')
        
        # BOLD Title and Ticks
        ax.set_title(f"Anchors: {f_obj['type']} o {g_obj['type']}", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            
        legend = ax.legend(loc='upper right', fontsize=16)
        plt.setp(legend.get_texts(), fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Graph 2: Inductive Point Cloud
        ax = axes[1]
        ax.scatter(X_test_np[:, L//2:].flatten(), ind_flat, c='red', s=15, alpha=0.6, label='Pred')
        ax.scatter(X_test_np[:, L//2:].flatten(), true_flat, c='black', s=5, alpha=0.2, label='True')
        
        # BOLD Title and Ticks
        ax.set_title(f"Naive Baseline\nMSE: {mse_ind:.4f}", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.set_ylim(-4, 4)
        
        legend = ax.legend(fontsize=16)
        plt.setp(legend.get_texts(), fontweight='bold')

        # Graph 3: RTE Point Cloud (Transductive)
        ax = axes[2]
        # ENSURE BLUE DOTS for Transductive
        ax.scatter(X_test_np[:, L//2:].flatten(), rte_flat, c='blue', s=15, alpha=0.6, label='Pred')
        ax.scatter(X_test_np[:, L//2:].flatten(), true_flat, c='black', s=5, alpha=0.2, label='True')
        
        # BOLD Title and Ticks
        ax.set_title(f"RTE \nMSE: {mse_rte:.4f}", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.set_ylim(-4, 4)
        
        legend = ax.legend(fontsize=16)
        plt.setp(legend.get_texts(), fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"compare_{eval_idx}.png"))
        plt.close()

def run_experiment(outdir):
    os.makedirs(outdir, exist_ok=True)
    lib = PrimitiveLibrary(n_per_family=3000, L=50) 
    lib.compute_task2vec_embeddings()
    
    # Instantiate Models
    d_rte = Decomposer(L=50, z_dim=16).to(DEVICE)
    composer = CompositionModel(L=50).to(DEVICE)
    inductive_model = InductiveBaseline(L=50).to(DEVICE)
    
    # Optimizers
    opt_rte = optim.Adam(d_rte.parameters(), lr=1e-3)
    opt_comp = optim.Adam(composer.parameters(), lr=1e-3)
    opt_ind = optim.Adam(inductive_model.parameters(), lr=1e-3)
    
    iters = 50000
    print(f"Training models for {iters} steps...")
    
    for it in range(1, iters+1):
        out = generate_batch_contrastive(lib, 64)
        X, Yf, Yg, Hh, Ht, Zf, Zg = out
        X_head = X[:, :25] # L=50 -> Head=25

        # 1. Train RTE Decomposer (MSE on Embeddings only)
        opt_rte.zero_grad()
        zn_f, zn_g = d_rte(X_head, Hh)
        loss_rte_decomp = nn.MSELoss()(zn_f, Zf) + nn.MSELoss()(zn_g, Zg)
        loss_rte_decomp.backward()
        opt_rte.step()
        
        # 2. Train Composer & Inductive Baseline
        opt_comp.zero_grad()
        loss_c = nn.MSELoss()(composer(X, Yf, Yg, Hh), Ht)
        loss_c.backward()
        opt_comp.step()
        
        opt_ind.zero_grad()
        loss_i = nn.MSELoss()(inductive_model(X, Hh), Ht)
        loss_i.backward()
        opt_ind.step()
        
        if it % 2000 == 0:
            print(f"Iter {it}: RTE_Decomp={loss_rte_decomp.item():.3f} | RTE_Comp={loss_c.item():.3f} | Inductive={loss_i.item():.3f}")

    compare_models(lib, d_rte, composer, inductive_model)
    visualize_comparison(outdir, lib, d_rte, composer, inductive_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out_compare_rte_vs_inductive")
    args = parser.parse_args()
    run_experiment(args.outdir)
