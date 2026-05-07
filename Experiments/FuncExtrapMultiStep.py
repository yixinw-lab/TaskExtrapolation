import argparse
import os
import random
import pickle
import math
from typing import Dict, Tuple, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global registry
CURRENT_PROBE = None
CURRENT_PCA = None
X_SPARSE_FIXED = torch.linspace(0, 10, 20).reshape(-1, 1).to(DEVICE)

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

class UniversalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([1, 64, 64, 1])
        
    def forward(self, x):
        return self.net(x)

# --- Gradient Embedding Logic ---

def pretrain_universal_probe(family_key: str, seed: int):
    print(f"Pre-training Universal Probe for {family_key}...")
    torch.manual_seed(seed)
    probe = UniversalProbe().to(DEVICE)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    rng = np.random.default_rng(seed)
    X_train, Y_train = [], []
    
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
    func, _ = sample_family_f_region(key, region, int(seed))
    raw_grad = compute_gradient_embedding(CURRENT_PROBE, func)
    z = CURRENT_PCA.transform(raw_grad.reshape(1, -1))[0]
    return z / (np.linalg.norm(z) + 1e-8)

# --- Function Family Definitions ---
PARAM_REGIONS = {
    'quadratic': {
        'a': {'F1_1': (0.5, 1.5), 'F1_2': (1.5, 2.5), 'F2': (2.5, 3.5), 
              'F3': (3.5, 4.5), 'F4': (4.5, 5.5), 'F5': (5.5, 6.5)}, 
        'b': {'F1_1': (-2.0, 2.0), 'F1_2': (-2.0, 2.0), 'F2': (-2.0, 2.0),
              'F3': (-2.0, 2.0), 'F4': (-2.0, 2.0), 'F5': (-2.0, 2.0)}, 
        'c': {'F1_1': (-2.0, 2.0), 'F1_2': (-2.0, 2.0), 'F2': (-2.0, 2.0),
              'F3': (-2.0, 2.0), 'F4': (-2.0, 2.0), 'F5': (-2.0, 2.0)}
    },
    'exp': {
        'alpha': {'F1_1': (1.0, 2.0), 'F1_2': (1.0, 2.0), 'F2': (1.0, 2.0),
                  'F3': (1.0, 2.0), 'F4': (1.0, 2.0), 'F5': (1.0, 2.0)}, 
        'beta':  {'F1_1': (0.05, 0.1), 'F1_2': (0.1, 0.15), 'F2': (0.15, 0.2),
                  'F3': (0.2, 0.25), 'F4': (0.25, 0.3), 'F5': (0.3, 0.35)}, 
        'c0': {'F1_1': (-1.0, 1.0), 'F1_2': (-1.0, 1.0), 'F2': (-1.0, 1.0),
               'F3': (-1.0, 1.0), 'F4': (-1.0, 1.0), 'F5': (-1.0, 1.0)}
    },
    'cubic': {
        'a3': {'F1_1': (0.5, 1.5), 'F1_2': (1.5, 2.5), 'F2': (2.5, 3.5),
               'F3': (3.5, 4.5), 'F4': (4.5, 5.5), 'F5': (5.5, 6.5)}, 
        'b3': {'F1_1': (2.0, 4.0), 'F1_2': (2.0, 4.0), 'F2': (2.0, 4.0),
               'F3': (2.0, 4.0), 'F4': (2.0, 4.0), 'F5': (2.0, 4.0)}, 
        'c3': {'F1_1': (-2.0, 2.0), 'F1_2': (-2.0, 2.0), 'F2': (-2.0, 2.0),
               'F3': (-2.0, 2.0), 'F4': (-2.0, 2.0), 'F5': (-2.0, 2.0)}
    },
    'tri_trend': {
        's': {'F1_1': (0.1, 0.2), 'F1_2': (0.2, 0.3), 'F2': (0.3, 0.4),
              'F3': (0.4, 0.5), 'F4': (0.5, 0.6), 'F5': (0.6, 0.7)}, 
        'A': {'F1_1': (0.5, 1.0), 'F1_2': (0.5, 1.0), 'F2': (0.5, 1.0),
              'F3': (0.5, 1.0), 'F4': (0.5, 1.0), 'F5': (0.5, 1.0)}
    },
    'sin_trend': {
        's2': {'F1_1': (0.1, 0.2), 'F1_2': (0.2, 0.3), 'F2': (0.3, 0.4),
               'F3': (0.4, 0.5), 'F4': (0.5, 0.6), 'F5': (0.6, 0.7)}, 
        'A2': {'F1_1': (0.5, 1.0), 'F1_2': (0.5, 1.0), 'F2': (0.5, 1.0),
               'F3': (0.5, 1.0), 'F4': (0.5, 1.0), 'F5': (0.5, 1.0)}
    }
}
KEY_PARAMS = {'quadratic': 'a', 'exp': 'beta', 'cubic': 'a3', 'tri_trend': 's', 'sin_trend': 's2'}

def sample_family_f_region(key: str, region: str, seed: int) -> Tuple[Callable, Dict]:
    rng = np.random.default_rng(seed); params = {}
    def sample_param(p_name):
        val = rng.uniform(PARAM_REGIONS[key][p_name][region][0], PARAM_REGIONS[key][p_name][region][1]); params[p_name] = val; return val
    if key == "quadratic": func = (lambda a,b,c: lambda x: a*x*x + b*x + c)(sample_param('a'), sample_param('b'), sample_param('c'))
    elif key == "exp": func = (lambda alpha,beta,c0: lambda x: alpha*(np.exp(beta*(x-10.0)) - 1.0) + c0)(sample_param('alpha'), sample_param('beta'), sample_param('c0'))
    elif key == "cubic": func = (lambda a3,b3,c3: lambda x: (a3/400.0)*(x-10.0)**3 + (b3/10.0)*(x-10.0) + c3)(sample_param('a3'), sample_param('b3'), sample_param('c3'))
    elif key == "tri_trend": P = 20.0/7.0; tri = lambda u: 2 * np.abs(2*((u/P) - np.floor(u/P))-1); func = (lambda s,A: lambda x: s*x + A*(tri(x)-0.5))(sample_param('s'), sample_param('A'))
    elif key == "sin_trend": P = 20.0/7.0; func = (lambda s2,A2: lambda x: s2*x + 0.5*A2*np.sin(2*np.pi*x/P))(sample_param('s2'), sample_param('A2'))
    else: raise ValueError(f"Unknown family key: {key}")
    return func, params

# --- Models ---

class FuncExtrapWithDynamicDistances(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: x(16) + y1(16) + y2(16) + y3_head(8) + d1(1) + d2(1) = 58
        self.block1 = mlp([58, 64, 64, 16]) 
        self.block2 = mlp([16, 64, 64, 16])
        self.out = mlp([16, 128, 8])
    def forward(self, x, y1, y2, y3_first_half, d1, d2, **kwargs):
        fin = torch.cat([x, y1, y2, y3_first_half, d1.view(-1, 1), d2.view(-1, 1)], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

class InductiveBaseline(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        # Inputs: x(L) + p3(1) + y3_head(L/2)
        input_dim = L + 1 + (L // 2)
        self.block1 = mlp([input_dim, 64, 64, 16])
        self.block2 = mlp([16, 64, 64, 16])
        self.out = mlp([16, 128, L // 2]) 

    def forward(self, x, p3, y3_first_half, **kwargs):
        fin = torch.cat([x, p3.view(-1, 1), y3_first_half], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

# === Sparse Baseline ===
class SparseBaseline(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        # Inputs: x(L) + y3_head(L/2)
        input_dim = L + (L // 2)
        self.block1 = mlp([input_dim, 64, 64, 16])
        self.block2 = mlp([16, 64, 64, 16])
        self.out = mlp([16, 128, L // 2])

    def forward(self, x, y3_first_half, **kwargs):
        fin = torch.cat([x, y3_first_half], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

def train_mse(net, loader, epochs=200, lr=1e-3, weight_decay=1e-4):
    net.to(DEVICE).train(); opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best = float('inf'); best_state = None; patience, bad = 40, 0
    for ep in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            yhat = net(**xb)
            loss = torch.mean((yhat - yb.to(DEVICE))**2)
            loss.backward(); opt.step(); total += loss.item()*yb.shape[0]; n += yb.shape[0]
        if total/n + 1e-8 < best:
            best = total/n; best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; bad=0
        else: bad += 1
        if bad > patience: break
    if best_state is not None: net.load_state_dict(best_state)

def build_dataset_with_distances(key: str, n_samples: int, L: int, ref_pool: List, f3_tasks: List, axis: np.ndarray, f_target_centroid: np.ndarray, mode: str, rng: np.random.Generator):
    X = rng.uniform(0.0, 20.0, size=(n_samples, L)).astype(np.float32)
    F1, F2, F3h, F3t, D1, D2, P3 = [], [], [], [], [], [], []
    
    local_cache = {}
    
    def get_e(r_key, r_reg, r_seed):
        k = (r_key, r_reg, r_seed)
        if k not in local_cache:
            local_cache[k] = get_embedding_pipeline(r_key, r_reg, r_seed)
        return local_cache[k]

    for i in range(n_samples):
        f1_seed, f1_region, f1_func = ref_pool[rng.integers(len(ref_pool))]
        f2_seed, f2_region, f2_func = ref_pool[rng.integers(len(ref_pool))]
        
        f3_seed, f3_region, f3_func = f3_tasks[rng.integers(len(f3_tasks))]
        
        x = X[i]; F1.append(f1_func(x).astype(np.float32)); F2.append(f2_func(x).astype(np.float32))
        y3 = f3_func(x).astype(np.float32); F3h.append(y3[:L//2]); F3t.append(y3[L//2:])
        
        e1 = get_e(key, f1_region, f1_seed)
        e2 = get_e(key, f2_region, f2_seed)
        p1 = np.dot(e1, axis); p2 = np.dot(e2, axis)
        
        e3 = get_e(key, f3_region, f3_seed)
        p3 = np.dot(e3, axis)

        D1.append(abs(p3 - p1)); D2.append(abs(p3 - p2))
        P3.append(p3)

    return (X, np.stack(F1), np.stack(F2), np.stack(F3h), np.stack(F3t), 
            np.array(D1, dtype=np.float32), np.array(D2, dtype=np.float32), 
            np.array(P3, dtype=np.float32))

def batch_loader(data_tuple, y, batch_size):
    N = y.shape[0]
    for i in range(0, N, batch_size):
        sl = slice(i, min(N, i + batch_size))
        xb = {
            "x": torch.as_tensor(data_tuple[0][sl], device=DEVICE), 
            "y1": torch.as_tensor(data_tuple[1][sl], device=DEVICE),
            "y2": torch.as_tensor(data_tuple[2][sl], device=DEVICE), 
            "y3_first_half": torch.as_tensor(data_tuple[3][sl], device=DEVICE),
            "d1": torch.as_tensor(data_tuple[5][sl], device=DEVICE), 
            "d2": torch.as_tensor(data_tuple[6][sl], device=DEVICE),
            "p3": torch.as_tensor(data_tuple[7][sl], device=DEVICE) 
        }
        yb = torch.as_tensor(y[sl], device=DEVICE); yield xb, yb

# --- Evaluation and Plotting ---

def plot_F2_comparison(out_png: str, key: str, f2_func: Callable, X_test: np.ndarray, Y_true: np.ndarray, predictions: Dict[str, np.ndarray], title_suffix: str):
    n_models = len(predictions)
    if n_models == 0: return
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten(); 
    # BOLD and INCREASE Title Font Size
    fig.suptitle(f"F2 Extrapolation ({title_suffix}): {key}", fontsize=30, fontweight='bold')
    
    xs_curve = np.linspace(0, 20, 2000); y_curve = f2_func(xs_curve)
    L = X_test.shape[1]; xs_tail = X_test[:, L//2:]
    
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]; mse = float(np.mean((y_pred - Y_true)**2))
        ax.plot(xs_curve, y_curve, '--', linewidth=2.0, alpha=0.95, color='black', label='True Function')
        ax.scatter(xs_tail.flatten(), Y_true.flatten(), s=25, alpha=0.5, c='tab:green', label='True Points')
        
        # COLOR LOGIC
        if "Transductive" in model_name or "RTE" in model_name:
            c_pred = 'blue'
        else:
            c_pred = 'tab:red'
            
        ax.scatter(xs_tail.flatten(), y_pred.flatten(), s=25, alpha=0.5, c=c_pred, label='Predicted')
        
        # BOLD and INCREASE Font Sizes for Axes, Titles, Ticks
        ax.set_title(f"{model_name}\nMSE = {mse:.3e}", fontsize=24, fontweight='bold')
        ax.set_xlabel("x", fontsize=22, fontweight='bold')
        
        # Increase Tick Size and make bold
        ax.tick_params(axis='both', which='major', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            
        ax.grid(alpha=0.3)
        
        if i == 0: 
            ax.set_ylabel("y", fontsize=22, fontweight='bold')
            # Increase Legend Size and Bold text
            legend = ax.legend(fontsize=16, frameon=False, loc='upper left')
            plt.setp(legend.get_texts(), fontweight='bold')
            
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(out_png, dpi=150); plt.close(fig); print(f"Saved: {out_png}")

def evaluate_multistep_extrapolation(model, key: str, target_region: str, n_steps: int, L: int, seed: int, axis: np.ndarray, c_f2: np.ndarray):
    rng = np.random.default_rng(seed)
    model.eval()
    
    # 1. Sample the distant target task
    f_target, _ = sample_family_f_region(key, target_region, seed)
    X_test = rng.uniform(0.0, 20.0, size=(1, L)).astype(np.float32)
    X_test = np.sort(X_test, axis=1) # Sort for sequential context
    
    Y_target_true = f_target(X_test[0]).astype(np.float32) # Full 16 points
    y_target_head = Y_target_true[:L//2]                   # First 8 points
    Y_target_true_tail = Y_target_true[L//2:]              # Last 8 points (for MSE)
    
    # 2. Sample the starting anchor from the boundary of the training set
    f_anc, _ = sample_family_f_region(key, 'F1_2', seed + 1)
    Y_anc_current = f_anc(X_test[0]).astype(np.float32)    # Full 16 points
    
    # 3. Get Embeddings and Projections
    e_anc = get_embedding_pipeline(key, 'F1_2', seed + 1)
    e_target = get_embedding_pipeline(key, target_region, seed)
    
    p_anc = np.dot(e_anc, axis)
    p_target = np.dot(e_target, axis)
    
    # Calculate step sizes for interpolation
    dp = (p_target - p_anc) / n_steps
    dy_head = (y_target_head - Y_anc_current[:L//2]) / n_steps
    
    # 4. Iterative Stepping
    current_y1 = Y_anc_current # Starts as 16 points
    current_p = p_anc
    
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            next_p = p_anc + (step * dp)
            ghost_y_head = Y_anc_current[:L//2] + (step * dy_head)
            d_step = abs(next_p - current_p)
            
            xb = {
                "x": torch.as_tensor(X_test, device=DEVICE),
                "y1": torch.as_tensor(current_y1.reshape(1, -1), device=DEVICE),
                "y2": torch.as_tensor(current_y1.reshape(1, -1), device=DEVICE), 
                "y3_first_half": torch.as_tensor(ghost_y_head.reshape(1, -1), device=DEVICE),
                "d1": torch.as_tensor(np.array([d_step], dtype=np.float32), device=DEVICE),
                "d2": torch.as_tensor(np.array([d_step], dtype=np.float32), device=DEVICE),
                "p3": torch.as_tensor(np.array([next_p], dtype=np.float32), device=DEVICE)
            }
            
            pred = model(**xb).cpu().numpy()[0]
            current_y1 = np.concatenate([ghost_y_head, pred])
            current_p = next_p
            
    mse = np.mean((pred - Y_target_true_tail)**2)
    return mse

def run_comparison(outdir: str, seed: int, n_train: int, n_test: int, L: int, epochs: int, batch_size: int, lr: float):
    os.makedirs(outdir, exist_ok=True); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    all_metrics = {}
    
    global CURRENT_PROBE, CURRENT_PCA 
    
    for key in KEY_PARAMS.keys():
        title = key.replace("_", " ").title()
        print(f"\n{'='*20}\nEXPERIMENT FOR: {title} [{key}]\n{'='*20}"); all_metrics[key] = {}
        
        CURRENT_PROBE = pretrain_universal_probe(key, seed)
        CURRENT_PCA = fit_pca_for_family(key, CURRENT_PROBE, seed)
        
        # Train systems
        (gt_trans, gt_ind, sparse_gt), axis_gt, c_f2_gt, ref_gt = run_experiment(key, n_train, L, seed, epochs, batch_size, lr, "GT")
        (t2v_trans, t2v_ind, sparse_t2v), axis_t2v, c_f2_t2v, ref_t2v = run_experiment(key, n_train, L, seed, epochs, batch_size, lr, "T2V")
        
        # Systems Registry
        systems = [
            ("Sparse Baseline", sparse_t2v, axis_t2v, c_f2_t2v, ref_t2v), 
            ("GT Transductive", gt_trans, axis_gt, c_f2_gt, ref_gt),
            ("GT Inductive",    gt_ind,    axis_gt, c_f2_gt, ref_gt),
            ("RTE",             t2v_trans, axis_t2v, c_f2_t2v, ref_t2v),
            ("T2V Inductive",   t2v_ind,   axis_t2v, c_f2_t2v, ref_t2v),
        ]

        print(f"\n--- Evaluating models on F2 test set (Monte Carlo 400 runs)... ---")
        rng_eval = np.random.default_rng(seed + 999)
        
        # Metrics storage
        mse_history = {name: [] for name, _, _, _, _ in systems}
        
        # Storage for the very last run for plotting
        last_run_data = {} # Stores (X_test, Y_true, f3_func)
        last_run_preds = {}

        N_MC = 400
        for i in tqdm(range(N_MC), desc="Monte Carlo Eval"):
            f3_test_seed = int(rng_eval.integers(1e9))
            f3_test_func, _ = sample_family_f_region(key, "F2", f3_test_seed)
            f3_test_tasks = [(f3_test_seed, "F2", f3_test_func)]
            
            # For each model, build data and predict
            for name, model, axis, c_f2, ref_pool in systems:
                rng_build = np.random.default_rng(seed + 999 + i)
                test_data = build_dataset_with_distances(key, n_test, L, ref_pool, f3_test_tasks, axis, c_f2, 'test', rng_build)
                Y_true, X_test = test_data[4], test_data[0]
                
                model.eval()
                with torch.no_grad():
                    loader_test = batch_loader(test_data, Y_true, batch_size)
                    pred = torch.cat([model(**xb) for xb, yb in loader_test]).cpu().numpy()
                
                mse = np.mean((pred - Y_true)**2)
                mse_history[name].append(mse)
                
                # Save data if this is the last iteration
                if i == N_MC - 1:
                    last_run_preds[name] = pred
                    last_run_data = (X_test, Y_true, f3_test_func)

        # Compute Statistics
        print("\n--- Monte Carlo Results ---")
        for name in mse_history:
            mses = np.array(mse_history[name])
            mean_mse = np.mean(mses)
            std_err = np.std(mses, ddof=1) / np.sqrt(len(mses))
            ci95 = 1.96 * std_err
            all_metrics[key][name] = {"mean": mean_mse, "ci": ci95}
            print(f"{name}: {mean_mse:.4e} +/- {ci95:.4e}")

        # === PLOTTING (Using data from the last MC run) ===
        X_final, Y_final, f_final = last_run_data
        
        # Filter: Exclude Sparse Baseline (STRICT FILTERING)
        preds_all = {k: v for k, v in last_run_preds.items() if "Sparse" not in k}
        
        plot_F2_comparison(
            os.path.join(outdir, f"{key}_All_Comparison.png"), 
            key, f_final, X_final, Y_final, preds_all, "All Models"
        )
        
        # === MULTI-STEP EVALUATION FOR ALL FAMILIES ===
        print("\n" + "*"*60)
        print(f"RUNNING MULTI-STEP EXTRAPOLATION EVALUATION FOR: {key.upper()}")
        print("*"*60)
        
        print(f"{'Model':<20} | {'F4 (3-Step MSE)':<15} | {'F5 (4-Step MSE)':<15}")
        print("-" * 55)
        
        for name, model, axis, c_f2, _ in systems:
            mse_f4 = evaluate_multistep_extrapolation(model, key, 'F4', n_steps=3, L=L, seed=seed+555, axis=axis, c_f2=c_f2)
            mse_f5 = evaluate_multistep_extrapolation(model, key, 'F5', n_steps=4, L=L, seed=seed+666, axis=axis, c_f2=c_f2)
            print(f"{name:<20} | {mse_f4:<15.4e} | {mse_f5:<15.4e}")
            
        print("*"*60 + "\n")
            
        CURRENT_PROBE = None
        CURRENT_PCA = None

    print("\n" + "="*70 + "\nFinal Results Summary (Mean MSE +/- 95% CI)\n" + "="*70)
    print(f"{'Family':<12}\t{'Model':<30}\t{'Result'}")
    print("-"*70)
    for key, metrics in all_metrics.items():
        for model_name, results in sorted(metrics.items(), key=lambda item: item[1]['mean']):
            print(f"{key:<12}\t{model_name:<30}\t{results['mean']:.3e} +/- {results['ci']:.3e}")
        print("-"*70)
    with open(os.path.join(outdir, "metrics_comparison.pkl"), "wb") as fh: pickle.dump(all_metrics, fh)

# --- Training Orchestration ---

def run_experiment(key: str, n_train_samples: int, L: int, base_seed: int, epochs: int, batch_size: int, lr: float, axis_type: str):
    print(f"\n--- Training: {axis_type} Axis for '{key}' ---")
    rng = np.random.default_rng(base_seed)
    
    # 1. Axis Discovery
    if axis_type == "GT":
        f1_1_seeds = rng.integers(1e9, size=50); f2_seeds = rng.integers(1e9, size=50)
        E_f1_1 = np.array([get_embedding_pipeline(key, "F1_1", s) for s in f1_1_seeds])
        E_f2 = np.array([get_embedding_pipeline(key, "F2", s) for s in f2_seeds])
        c_f1_1 = E_f1_1.mean(axis=0); c_f2 = E_f2.mean(axis=0)
        axis = c_f2 - c_f1_1; axis /= np.linalg.norm(axis)
    elif axis_type == "T2V":
        f1_1_seeds = rng.integers(1e9, size=50); f1_2_seeds = rng.integers(1e9, size=50); f2_seeds = rng.integers(1e9, size=100)
        E_f1_1 = np.array([get_embedding_pipeline(key, "F1_1", s) for s in tqdm(f1_1_seeds, desc="Embed F1_1")])
        E_f1_2 = np.array([get_embedding_pipeline(key, "F1_2", s) for s in tqdm(f1_2_seeds, desc="Embed F1_2")])
        E_f2 = np.array([get_embedding_pipeline(key, "F2", s) for s in tqdm(f2_seeds, desc="Embed F2")])
        E_train = np.vstack([E_f1_1, E_f1_2])
        c_train = E_train.mean(axis=0); c_f2 = E_f2.mean(axis=0)
        axis = c_f2 - c_train; axis /= np.linalg.norm(axis)
        
    # 2. Training Data Generation
    ref_pool = [(s, "F1_1", sample_family_f_region(key, "F1_1", s)[0]) for s in rng.integers(1e9, size=4)]
    f3_train_tasks = [(s, "F1_1", sample_family_f_region(key, "F1_1", s)[0]) for s in rng.integers(1e9, size=50)] + \
                     [(s, "F1_2", sample_family_f_region(key, "F1_2", s)[0]) for s in rng.integers(1e9, size=50)]
    random.shuffle(f3_train_tasks)
    
    data_tuple = build_dataset_with_distances(key, n_train_samples, L, ref_pool, f3_train_tasks, axis, c_f2, 'train', rng)
    loader = list(batch_loader(data_tuple, data_tuple[4], batch_size))
    
    # 3. Train Transductive Model
    print(f"Training Transductive ({axis_type})...")
    model_trans = FuncExtrapWithDynamicDistances().to(DEVICE)
    train_mse(model_trans, loader, epochs=epochs, lr=lr)
    
    # 4. Train Inductive Model
    print(f"Training Inductive ({axis_type})...")
    model_ind = InductiveBaseline(L=L).to(DEVICE)
    train_mse(model_ind, loader, epochs=epochs, lr=lr)
    
    # 5. Train Sparse Baseline
    print(f"Training Sparse Baseline...")
    model_sparse = SparseBaseline(L=L).to(DEVICE)
    train_mse(model_sparse, loader, epochs=epochs, lr=lr)

    test_ref_pool = [(s, "F1_2", sample_family_f_region(key, "F1_2", s)[0]) for s in rng.integers(1e9, size=4)]
    
    # Return all 3 models
    return (model_trans, model_ind, model_sparse), axis, c_f2, test_ref_pool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_fair_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train_samples", type=int, default=20000)
    parser.add_argument("--n_test_samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run_comparison(outdir=args.outdir, seed=args.seed, n_train=args.n_train_samples, n_test=args.n_test_samples, L=16, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

if __name__ == "__main__": main()
