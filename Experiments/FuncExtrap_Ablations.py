import argparse
import os
import random
import pickle
import math
import pandas as pd # Added for CSV export
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Callable, List

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global registry for current embedding state
CURRENT_PROBE = None
CURRENT_PCA = None
X_SPARSE_FIXED = torch.linspace(0, 10, 20).reshape(-1, 1).to(DEVICE)

# ==========================================
# 1. Neural Network Helpers & Probe
# ==========================================

def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)

class UniversalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([1, 64, 64, 1])
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Efficient Embedding Engine (Pre-computation)
# ==========================================

def pretrain_universal_probe(family_key: str, seed: int):
    print(f"Pre-training Universal Probe for {family_key}...")
    torch.manual_seed(seed)
    probe = UniversalProbe().to(DEVICE)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    rng = np.random.default_rng(seed)
    X_train, Y_train = [], []
    
    # Train on mix of F1_1 and F1_2
    for region in ['F1_1', 'F1_2']:
        for _ in range(50):
            s = int(rng.integers(1e9))
            f, _ = sample_family_f_region(family_key, region, s)
            xs = rng.uniform(0, 20, 50).astype(np.float32)
            ys = f(xs).astype(np.float32)
            X_train.append(xs)
            Y_train.append(ys)
            
    X_t = torch.from_numpy(np.concatenate(X_train).reshape(-1, 1)).to(DEVICE)
    Y_t = torch.from_numpy(np.concatenate(Y_train).reshape(-1, 1)).to(DEVICE)
    
    probe.train()
    for _ in range(500):
        opt.zero_grad()
        pred = probe(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()
    return probe

def fit_pca_for_family(family_key: str, probe: nn.Module, seed: int):
    print(f"Fitting PCA for {family_key} gradients...")
    rng = np.random.default_rng(seed)
    raw_grads = []
    
    for region in ['F1_1', 'F1_2']:
        for _ in range(50):
            s = int(rng.integers(1e9))
            f, _ = sample_family_f_region(family_key, region, s)
            grad = compute_gradient_embedding(probe, f)
            raw_grads.append(grad)
            
    pca = PCA(n_components=16) 
    pca.fit(np.stack(raw_grads))
    return pca

def compute_gradient_embedding(probe: nn.Module, f: Callable) -> np.ndarray:
    probe.zero_grad()
    y_true = f(X_SPARSE_FIXED.cpu().numpy().flatten()).astype(np.float32)
    y_true_t = torch.from_numpy(y_true).reshape(-1, 1).to(DEVICE)
    y_pred = probe(X_SPARSE_FIXED)
    loss = nn.MSELoss()(y_pred, y_true_t)
    loss.backward()
    grads = []
    for p in probe.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1).cpu().numpy())
        else:
            grads.append(np.zeros(p.numel()))
    return np.concatenate(grads)

def get_embedding_pipeline(key, region, seed):
    # Optimized: In a real run, you'd cache this heavily, but for this script 
    # it's fast enough because we only call it for the dataset generation steps.
    func, _ = sample_family_f_region(key, region, int(seed))
    raw_grad = compute_gradient_embedding(CURRENT_PROBE, func)
    z = CURRENT_PCA.transform(raw_grad.reshape(1, -1))[0]
    return z / (np.linalg.norm(z) + 1e-8)

# ==========================================
# 3. Domain Logic (Functions)
# ==========================================
PARAM_REGIONS = {
    'quadratic': {
        'a': {'F1_1': (0.5, 1.5), 'F1_2': (1.5, 2.5), 'F2': (2.5, 3.5)}, 
        'b': {'F1_1': (-2.0, 2.0), 'F1_2': (-2.0, 2.0), 'F2': (-2.0, 2.0)}, 
        'c': {'F1_1': (-2.0, 2.0), 'F1_2': (-2.0, 2.0), 'F2': (-2.0, 2.0)}
    }
}

def sample_family_f_region(key: str, region: str, seed: int) -> Tuple[Callable, Dict]:
    rng = np.random.default_rng(seed); params = {}
    
    # Only implemented quadratic for this ablation script as per your snippet
    if key == "quadratic":
        def sample_param(p_name):
            val = rng.uniform(PARAM_REGIONS[key][p_name][region][0], PARAM_REGIONS[key][p_name][region][1])
            params[p_name] = val
            return val
        
        a, b, c = sample_param('a'), sample_param('b'), sample_param('c')
        func = lambda x: a*x*x + b*x + c
        return func, params
    else:
        # Fallback for other families if you expand later
        return lambda x: x, {}

# ==========================================
# 4. Models
# ==========================================

class FuncTransducer(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches original snippet architecture
        # Inputs: x(16) + y1(16) + y2(16) + y3_head(8) + d1(1) + d2(1) = 58
        self.block1 = mlp([58, 64, 64, 16]) 
        self.block2 = mlp([16, 64, 64, 16])
        self.out = mlp([16, 128, 8])
    def forward(self, x, y1, y2, y3h, d1, d2):
        fin = torch.cat([x, y1, y2, y3h, d1.view(-1, 1), d2.view(-1, 1)], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

class InductiveBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: x(16) + y3_head(8) = 24
        self.block1 = mlp([24, 64, 64, 16])
        self.block2 = mlp([16, 64, 64, 16])
        self.out = mlp([16, 128, 8]) 
    def forward(self, x, y3h):
        fin = torch.cat([x, y3h], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

# ==========================================
# 5. Data & Training Ops
# ==========================================

def build_dataset_for_ablation(key, n_samples, L, ref_pool, f3_tasks, axis, mode, rng):
    """
    Constructs dataset efficiently. Returns numpy arrays.
    """
    X = rng.uniform(0.0, 20.0, size=(n_samples, L)).astype(np.float32)
    F1, F2, F3h, F3t, D1, D2 = [], [], [], [], [], []
    
    # Cache embeddings to speed up generation
    memo_embed = {}
    def get_e_cached(r_key, r_reg, r_seed):
        k = (r_reg, r_seed)
        if k not in memo_embed:
            memo_embed[k] = get_embedding_pipeline(r_key, r_reg, r_seed)
        return memo_embed[k]

    for i in range(n_samples):
        # Sample Anchors
        f1_seed, f1_region, f1_func = ref_pool[rng.integers(len(ref_pool))]
        f2_seed, f2_region, f2_func = ref_pool[rng.integers(len(ref_pool))]
        
        # Sample Task
        f3_seed, f3_region, f3_func = f3_tasks[rng.integers(len(f3_tasks))]
        
        # Function Evaluations
        x = X[i]
        F1.append(f1_func(x).astype(np.float32))
        F2.append(f2_func(x).astype(np.float32))
        y3 = f3_func(x).astype(np.float32)
        F3h.append(y3[:L//2])
        F3t.append(y3[L//2:]) # Targets (extrapolation)
        
        # Embeddings & Distances
        e1 = get_e_cached(key, f1_region, f1_seed)
        e2 = get_e_cached(key, f2_region, f2_seed)
        e3 = get_e_cached(key, f3_region, f3_seed)
        
        p1 = np.dot(e1, axis)
        p2 = np.dot(e2, axis)
        p3 = np.dot(e3, axis)

        D1.append(p3 - p1) # Signed distance, matching original logic usually
        D2.append(p3 - p2)

    return (X, np.stack(F1), np.stack(F2), np.stack(F3h), np.stack(F3t), 
            np.array(D1, dtype=np.float32), np.array(D2, dtype=np.float32))

def train_model(net, data_tuple, epochs=200, batch_size=4096):
    net.train()
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    X, Y1, Y2, Y3h, Y3t, D1, D2 = data_tuple
    
    N = X.shape[0]
    
    for _ in range(epochs):
        # Simple shuffling
        perm = torch.randperm(N)
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            
            x_b = torch.tensor(X[idx]).to(DEVICE)
            y3h_b = torch.tensor(Y3h[idx]).to(DEVICE)
            y3t_b = torch.tensor(Y3t[idx]).to(DEVICE) # Target
            
            if isinstance(net, InductiveBaseline):
                pred = net(x_b, y3h_b)
            else:
                y1_b = torch.tensor(Y1[idx]).to(DEVICE)
                y2_b = torch.tensor(Y2[idx]).to(DEVICE)
                d1_b = torch.tensor(D1[idx]).to(DEVICE)
                d2_b = torch.tensor(D2[idx]).to(DEVICE)
                pred = net(x_b, y1_b, y2_b, y3h_b, d1_b, d2_b)
                
            loss = nn.MSELoss()(pred, y3t_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

# ==========================================
# 6. Ablation Runner
# ==========================================

def run_ablation_experiment():
    key = "quadratic"
    L = 16
    seed = 42
    
    print(f"--- Setting up Experiment for {key} ---")
    global CURRENT_PROBE, CURRENT_PCA
    CURRENT_PROBE = pretrain_universal_probe(key, seed)
    CURRENT_PCA = fit_pca_for_family(key, CURRENT_PROBE, seed)
    
    rng = np.random.default_rng(seed)
    
    # --- 1. Train the Main "T2V" Model ---
    # Axis Discovery (T2V Style)
    print("Discovering Axis...")
    f1_1_seeds = rng.integers(1e9, size=50)
    f2_seeds = rng.integers(1e9, size=50)
    
    E_f1_1 = np.array([get_embedding_pipeline(key, "F1_1", s) for s in f1_1_seeds])
    E_f2 = np.array([get_embedding_pipeline(key, "F2", s) for s in f2_seeds])
    axis = np.mean(E_f2, axis=0) - np.mean(E_f1_1, axis=0)
    axis /= np.linalg.norm(axis)
    
    # Training Data
    print("Generating Training Data...")
    # Reference pool: 2 from F1_1, 2 from F1_2
    ref_pool_train = [(s, "F1_1", sample_family_f_region(key, "F1_1", s)[0]) for s in rng.integers(1e9, size=2)] + \
                     [(s, "F1_2", sample_family_f_region(key, "F1_2", s)[0]) for s in rng.integers(1e9, size=2)]
    
    # Tasks: 50 from F1_1, 50 from F1_2
    train_tasks = [(s, "F1_1", sample_family_f_region(key, "F1_1", s)[0]) for s in rng.integers(1e9, size=50)] + \
                  [(s, "F1_2", sample_family_f_region(key, "F1_2", s)[0]) for s in rng.integers(1e9, size=50)]
    
    data_train = build_dataset_for_ablation(key, 20000, L, ref_pool_train, train_tasks, axis, 'train', rng)
    
    trans_model = FuncTransducer().to(DEVICE)
    ind_model = InductiveBaseline().to(DEVICE)
    
    print("Training Transductive Model...")
    train_model(trans_model, data_train)
    
    print("Training Inductive Baseline...")
    train_model(ind_model, data_train)
    
    # --- 2. Run Ablations ---
    results = []
    
    def eval_scenario(name, model, anchor_mode='near', distance_mode='normal'):
        """
        anchor_mode: 'near' (F1_2), 'far' (F1_1), 'random' (Mixed)
        distance_mode: 'normal', 'zero', 'random'
        """
        model.eval()
        errs = []
        
        # MC Evaluation settings
        N_MC = 500
        f3_test_tasks = [(int(rng.integers(1e9)), "F2", sample_family_f_region(key, "F2", int(rng.integers(1e9)))[0]) 
                         for _ in range(N_MC)]
        
        # Determine Anchors
        if anchor_mode == 'near':
            pool = [(s, "F1_2", sample_family_f_region(key, "F1_2", s)[0]) for s in rng.integers(1e9, size=4)]
        elif anchor_mode == 'far':
            pool = [(s, "F1_1", sample_family_f_region(key, "F1_1", s)[0]) for s in rng.integers(1e9, size=4)]
        elif anchor_mode == 'random':
            pool = [(s, "F1_1", sample_family_f_region(key, "F1_1", s)[0]) for s in rng.integers(1e9, size=2)] + \
                   [(s, "F1_2", sample_family_f_region(key, "F1_2", s)[0]) for s in rng.integers(1e9, size=2)]
        
        # Build Data
        # We process all MC evals in one giant batch for speed
        test_data = build_dataset_for_ablation(key, N_MC, L, pool, f3_test_tasks, axis, 'test', rng)
        X, Y1, Y2, Y3h, Y3t, D1, D2 = test_data
        
        # Modify Distances if needed
        if distance_mode == 'zero':
            D1 = np.zeros_like(D1)
            D2 = np.zeros_like(D2)
        elif distance_mode == 'random':
            D1 = np.random.randn(*D1.shape).astype(np.float32)
            D2 = np.random.randn(*D2.shape).astype(np.float32)
            
        # Forward Pass
        with torch.no_grad():
            x_b = torch.tensor(X).to(DEVICE)
            y3h_b = torch.tensor(Y3h).to(DEVICE)
            
            if isinstance(model, InductiveBaseline):
                pred = model(x_b, y3h_b)
            else:
                y1_b = torch.tensor(Y1).to(DEVICE)
                y2_b = torch.tensor(Y2).to(DEVICE)
                d1_b = torch.tensor(D1).to(DEVICE)
                d2_b = torch.tensor(D2).to(DEVICE)
                pred = model(x_b, y1_b, y2_b, y3h_b, d1_b, d2_b)
            
            y_tgt = torch.tensor(Y3t).to(DEVICE)
            mse = torch.mean((pred - y_tgt)**2, dim=1).cpu().numpy()
            errs.extend(mse)
            
        mean_mse = np.mean(errs)
        ci = 1.96 * np.std(errs) / np.sqrt(len(errs))
        print(f"[{name}] MSE: {mean_mse:.4f} +/- {ci:.4f}")
        return {"Method": name, "MSE": mean_mse, "CI": ci}

    print("\n--- Running Final Ablations ---")
    
    # 1. Inductive Baseline
    results.append(eval_scenario("Inductive Baseline", ind_model))
    
    # 2. Standard Transductive (Near Anchors)
    results.append(eval_scenario("Standard Transductive (Near Anchors)", trans_model, anchor_mode='near'))
    
    # 3. Ablation: Far Anchors
    results.append(eval_scenario("Ablation: Far Anchors (F1_1)", trans_model, anchor_mode='far'))
    
    # 4. Ablation: Random Anchors
    results.append(eval_scenario("Ablation: Random Anchors", trans_model, anchor_mode='random'))
    
    # 5. Ablation: Zero Shift
    results.append(eval_scenario("Ablation: Zero Shift", trans_model, anchor_mode='near', distance_mode='zero'))
    
    # 6. Ablation: Random Axis (Noise)
    results.append(eval_scenario("Ablation: Random Axis", trans_model, anchor_mode='near', distance_mode='random'))
    
    # Save
    df = pd.DataFrame(results)
    print("\nResults Table:")
    print(df)
    df.to_csv("func_ablation_final.csv", index=False)
    print("Saved to func_ablation_final.csv")

if __name__ == "__main__":
    run_ablation_experiment()
