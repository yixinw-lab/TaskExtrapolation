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
# Task2Vec Encoder (Generic)
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
        self.probe.train()
        opt = optim.Adam(self.probe.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n_funcs = len(library.dense_curves)
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
        print("Fitting PCA...")
        raw_grads = [self._get_raw_gradients(i, library) for i in range(len(library.dense_curves))]
        raw_grads = np.stack(raw_grads)
        self.embeddings = self.pca.fit_transform(raw_grads)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        return self.embeddings

# --- 1. The Polynomial Library ---

class PolyLibrary:
    def __init__(self, n_per_degree=1000, max_degree=8, L=16, seed=42):
        self.rng = np.random.default_rng(seed)
        self.L = L
        self.x_range = np.linspace(-5, 5, L).astype(np.float32)
        self.functions = []  
        self.dense_curves = []
        
        print(f"Populating Library (Degrees 1-{max_degree}, {n_per_degree}/deg)...")
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
                for p, val in enumerate(c):
                    y += val * (x ** p)
                return y
            self.functions.append({'func': f, 'degree': degree, 'coeffs': c})
            self.dense_curves.append(f(self.x_range))

    def compute_task2vec_embeddings(self):
        self.encoder = Task2VecEncoder(self.x_range, z_dim=16)
        self.embeddings = self.encoder.fit_transform_library(self)
        self.nn_engine = NearestNeighbors(metric='euclidean')
        self.nn_engine.fit(self.embeddings)

# --- 2. Models ---

class LengthDecomposer(nn.Module):
    def __init__(self, L=16, z_dim=16):
        super().__init__()
        n_obs = L // 2
        in_dim = n_obs * 2 
        self.shared = mlp([in_dim, 256, 128, 64])
        self.head_z = nn.Linear(64, z_dim)
        self.head_c = nn.Linear(64, 1)
        
    def forward(self, x, h):
        sort_idx = torch.argsort(x, dim=1)
        x_sorted = torch.gather(x, 1, sort_idx)
        h_sorted = torch.gather(h, 1, sort_idx)
        fin = torch.cat([x_sorted, h_sorted], dim=-1)
        feat = self.shared(fin)
        return self.head_z(feat), self.head_c(feat)

class LengthExtender(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        # Input: x(L) + y_prev(L) + c(1)
        input_dim = L + L + 1
        self.block1 = mlp([input_dim, 256, 256, 128]) 
        self.block2 = mlp([128, 128, 64])
        self.out = mlp([64, 64, L]) 

    def forward(self, x, y_prev, c_new):
        fin = torch.cat([x, y_prev, c_new], dim=-1)
        return self.out(self.block2(self.block1(fin)))

class NaiveBaseline(nn.Module):
    def __init__(self, L=16):
        super().__init__()
        input_dim = L + (L // 2)
        self.block1 = mlp([input_dim, 256, 256, 128])
        # FIX: Input dim was 16, but block1 outputs 128. Changed to 128.
        self.block2 = mlp([128, 128, 64]) 
        self.out = mlp([64, 64, L // 2]) # Predicts Tail

    def forward(self, x, h_head):
        fin = torch.cat([x, h_head], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

# --- 3. Data Generation ---

def generate_batch(library, batch_size, max_degree=8, fixed_degree=None):
    L = library.L
    inds = np.arange(len(library.functions))
    
    batch_X = []
    batch_Y_prev, batch_Y_target = [], []
    batch_H_head, batch_H_tail = [], []
    batch_Z_prev = []
    batch_C_new = []
    
    x_query = library.rng.uniform(-5.0, 5.0, size=(batch_size, L)).astype(np.float32)
    
    for i in range(batch_size):
        # 1. Choose Predecessor P_{N-1}
        if fixed_degree is not None:
            deg_prev = fixed_degree - 1
            candidates = [idx for idx, f in enumerate(library.functions) if f['degree'] == deg_prev]
            idx_prev = np.random.choice(candidates) if candidates else np.random.choice(inds)
        else:
            idx_prev = np.random.choice(inds)
            
        f_prev = library.functions[idx_prev]
        z_prev = library.embeddings[idx_prev]
        deg_prev = f_prev['degree']
        deg_target = deg_prev + 1
        
        # 2. Sample new coefficient
        scale = 1.0 / (5.0 ** deg_target)
        c_val = library.rng.uniform(-2.0, 2.0) * scale
        
        # 3. Compute Target (Ground Truth)
        y_prev_query = f_prev['func'](x_query[i])
        term_new = c_val * (x_query[i] ** deg_target)
        y_target_query = y_prev_query + term_new
        
        # Normalized coefficient
        c_norm = c_val * (5.0 ** deg_target)
        
        batch_X.append(x_query[i])
        batch_Y_prev.append(y_prev_query) 
        batch_Y_target.append(y_target_query)
        batch_H_head.append(y_target_query[:L//2])
        batch_H_tail.append(y_target_query[L//2:])
        batch_Z_prev.append(z_prev)
        batch_C_new.append(np.array([c_norm], dtype=np.float32))
        
    return (
        torch.tensor(np.stack(batch_X)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Y_prev)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Y_target)).float().to(DEVICE),
        torch.tensor(np.stack(batch_H_head)).float().to(DEVICE),
        torch.tensor(np.stack(batch_H_tail)).float().to(DEVICE),
        torch.tensor(np.stack(batch_Z_prev)).float().to(DEVICE),
        torch.tensor(np.stack(batch_C_new)).float().to(DEVICE)
    )

# --- 4. Neural Re-Ranking Logic ---

def select_best_candidate_neural(library, extender_model, z_pred, c_pred, h_head_true, x_query_full, k=5):
    """
    Neural Re-Ranking:
    1. Retrieve k neighbors (candidate predecessors).
    2. Feed (x, candidate, c_pred) into the LEARNED EXTENDER.
    3. Compare Extender's predicted HEAD with Ground Truth HEAD.
    """
    # 1. Retrieval
    _, indices = library.nn_engine.kneighbors(z_pred.reshape(1, -1), n_neighbors=k)
    indices = indices[0]
    candidate_funcs = [library.functions[i]['func'] for i in indices]
    
    best_error = float('inf')
    best_y_prev_on_query = None
    
    x_in = torch.tensor(x_query_full).float().to(DEVICE).unsqueeze(0) # (1, L)
    c_in = c_pred.reshape(1, 1).to(DEVICE) # (1, 1)
    L = x_query_full.shape[0]

    for func in candidate_funcs:
        y_prev_np = func(x_query_full)
        y_prev_in = torch.tensor(y_prev_np).float().to(DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            y_full_pred = extender_model(x_in, y_prev_in, c_in)
            
        y_head_pred = y_full_pred[0, :L//2]
        mse = nn.MSELoss()(y_head_pred, h_head_true).item()
        
        if mse < best_error:
            best_error = mse
            best_y_prev_on_query = y_prev_np
            
    return best_y_prev_on_query

# --- 5. Experiment Runners ---

def visualize_results(outdir, library, decomposer, extender, baseline):
    decomposer.eval(); extender.eval(); baseline.eval()
    print("\nGenerating Dense Visualization for Degree 9 Extrapolation (10 Samples)...")

    # === LOOP 10 TIMES TO GENERATE 10 IMAGES ===
    for img_idx in range(10):
        # --- 1. Setup ONE Fixed Task (Degree 9) for Plotting ---
        deg_prev = 8
        deg_target = 9
        
        idx_prev = np.random.randint(len(library.functions))
        while library.functions[idx_prev]['degree'] != deg_prev:
            idx_prev = np.random.randint(len(library.functions))
            
        f_prev_func = library.functions[idx_prev]['func']
        scale = 1.0 / (5.0 ** deg_target)
        c_val = np.random.uniform(-2.0, 2.0) * scale
        c_norm = c_val * (5.0 ** deg_target)
        
        def f_target_func(x):
            return f_prev_func(x) + c_val * (x**deg_target)

        # --- 2. Generate Dense Ground Truth Curves (for plotting lines) ---
        xs_curve = np.linspace(-5, 5, 1000).astype(np.float32)
        ys_prev_curve = f_prev_func(xs_curve)
        ys_target_curve = f_target_func(xs_curve)

        # --- 3. Run Inference on Many Random Batches to generate "Clouds" ---
        n_batches = 500  # Increased for denser clouds
        L = library.L
        
        all_x_tail = []
        preds_base = []
        preds_rte = []
        
        first_retrieved_curve = None
        
        with torch.no_grad():
            for i in range(n_batches):
                x_np = np.random.uniform(-5, 5, size=(1, L)).astype(np.float32)
                y_tgt = f_target_func(x_np)
                
                X = torch.tensor(x_np).float().to(DEVICE)
                H_head = torch.tensor(y_tgt[:, :L//2]).float().to(DEVICE)
                
                # A. Baseline
                p_b = baseline(X, H_head).cpu().numpy()
                preds_base.append(p_b.flatten())
                
                # B. Neural RTE
                z_p, c_p = decomposer(X[:, :L//2], H_head)
                best_y_prev = select_best_candidate_neural(
                    library, extender, z_p.cpu().numpy()[0], c_p.cpu(), 
                    H_head[0], x_np[0], k=5
                )
                
                if i == 0:
                    # Capture the retrieved predecessor for the very first batch to plot
                    _, indices = library.nn_engine.kneighbors(z_p.cpu().numpy(), n_neighbors=5)
                    top_idx = indices[0][0]
                    first_retrieved_curve = library.functions[top_idx]['func'](xs_curve)

                y_in = torch.tensor(best_y_prev).float().to(DEVICE).unsqueeze(0)
                full_pred = extender(X, y_in, c_p)
                p_r = full_pred[0, L//2:].cpu().numpy()
                preds_rte.append(p_r.flatten())
                
                all_x_tail.append(x_np[0, L//2:])

        flat_x = np.concatenate(all_x_tail)
        flat_base = np.concatenate(preds_base)
        flat_rte = np.concatenate(preds_rte)
        
        flat_gt = f_target_func(flat_x)
        mse_base = np.mean((flat_base - flat_gt)**2)
        mse_rte = np.mean((flat_rte - flat_gt)**2)

        # --- 4. Plotting (Matching eval_4.png style) ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # Panel 1: Decomposition (Predecessor Recovery)
        ax = axes[0]
        ax.set_title(f"Decomposition (Sample {img_idx+1})", fontsize=24, fontweight='bold')
        
        # Plot True vs Retrieved Predecessor
        ax.plot(xs_curve, ys_prev_curve, color='blue', linestyle='-', lw=2, label='True P_prev')
        if first_retrieved_curve is not None:
            ax.plot(xs_curve, first_retrieved_curve, color='blue', linestyle='--', alpha=0.6, lw=2, label='Retrieved P_prev')
        
        # Plot True Target (BOLDED)
        ax.plot(xs_curve, ys_target_curve, color='green', linestyle='-', lw=5, label='True Target (P_new)')
        
        ax.legend(loc='upper left', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(False) # Clean look like image
        ax.set_ylim(min(ys_prev_curve.min(), ys_target_curve.min())-1, 
                    max(ys_prev_curve.max(), ys_target_curve.max())+1)

        # Panel 2: Baseline (Scatter Cloud)
        ax = axes[1]
        ax.set_title(f"Naive Baseline (MSE: {mse_base:.4f})", fontsize=24, fontweight='bold')
        
        # Plot Scatter
        ax.scatter(flat_x, flat_base, s=6, c='red', alpha=0.4, label='Preds')
        # Plot Smooth Ground Truth Overlay (BOLDED)
        ax.plot(xs_curve, ys_target_curve, color='black', alpha=0.9, lw=4, label='True Target')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(axes[0].get_ylim())
        # No grid, cleaner look

        # Panel 3: Neural RTE (Scatter Cloud) - RENAMED from RAG
        ax = axes[2]
        ax.set_title(f"RTE (MSE: {mse_rte:.4f})", fontsize=24, fontweight='bold')
        
        # Plot Scatter (Blue)
        ax.scatter(flat_x, flat_rte, s=6, c='tab:blue', alpha=0.4, label='Preds')
        # Plot Smooth Ground Truth Overlay (BOLDED)
        ax.plot(xs_curve, ys_target_curve, color='black', alpha=0.9, lw=4, label='True Target')
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        ax.set_ylim(axes[0].get_ylim())

        plt.tight_layout()
        save_path = os.path.join(outdir, f"dense_neural_extrap_{img_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved dense visualization to {save_path}")

def run_experiment(outdir):
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Build Library (Degrees 1-8)
    lib = PolyLibrary(n_per_degree=1000, max_degree=8, L=16)
    lib.compute_task2vec_embeddings()
    
    # 2. Models
    decomposer = LengthDecomposer(L=16, z_dim=16).to(DEVICE)
    extender = LengthExtender(L=16).to(DEVICE)
    baseline = NaiveBaseline(L=16).to(DEVICE)
    
    opt_decomp = optim.Adam(decomposer.parameters(), lr=1e-3)
    opt_ext = optim.Adam(extender.parameters(), lr=1e-3)
    opt_base = optim.Adam(baseline.parameters(), lr=1e-3)
    
    # 3. Training
    iters = 20000
    print(f"\nTraining for {iters} iterations...")
    
    for it in range(1, iters+1):
        out = generate_batch(lib, batch_size=64, max_degree=8)
        X, Y_prev, Y_target, H_head, H_tail, Z_prev_true, C_true = out
        
        # Train Decomposer
        opt_decomp.zero_grad()
        z_pred, c_pred = decomposer(X[:, :8], H_head) 
        loss_d = nn.MSELoss()(z_pred, Z_prev_true) + nn.MSELoss()(c_pred, C_true)
        loss_d.backward()
        opt_decomp.step()
        
        # Train Extender
        opt_ext.zero_grad()
        full_pred = extender(X, Y_prev, C_true)
        loss_ext = nn.MSELoss()(full_pred, Y_target)
        loss_ext.backward()
        opt_ext.step()
        
        # Train Baseline
        opt_base.zero_grad()
        base_pred = baseline(X, H_head)
        loss_base = nn.MSELoss()(base_pred, H_tail)
        loss_base.backward()
        opt_base.step()
        
        if it % 2000 == 0:
            print(f"Iter {it}: Decomp={loss_d.item():.4f} | Extender={loss_ext.item():.4f} | Base={loss_base.item():.4f}")

    # 4. Evaluation with Confidence Intervals (Monte Carlo over Tasks)
    print("\n--- Evaluation on 50 MC Tasks (Degree 9) ---")
    
    decomposer.eval(); extender.eval(); baseline.eval()
    
    n_eval_tasks = 50
    task_mses_base = []
    task_mses_rte = []
    
    for t_idx in range(n_eval_tasks):
        # 4a. Create a specific Degree 9 Task
        deg_prev = 8
        deg_target = 9
        
        idx_prev = np.random.randint(len(lib.functions))
        while lib.functions[idx_prev]['degree'] != deg_prev:
            idx_prev = np.random.randint(len(lib.functions))
        
        f_prev = lib.functions[idx_prev]
        # Sample coeff
        scale = 1.0 / (5.0 ** deg_target)
        c_val = np.random.uniform(-2.0, 2.0) * scale
        c_norm = c_val * (5.0 ** deg_target)
        
        # 4b. Sample a batch of points for this SPECIFIC task to evaluate it
        # (Simulating a test set for this task)
        eval_batch_size = 50
        x_query = np.random.uniform(-5.0, 5.0, size=(eval_batch_size, lib.L)).astype(np.float32)
        
        # Ground Truths
        y_prev_q = f_prev['func'](x_query)
        term_new = c_val * (x_query ** deg_target)
        y_target_q = y_prev_q + term_new
        
        X_t = torch.tensor(x_query).float().to(DEVICE)
        H_head_t = torch.tensor(y_target_q[:, :8]).float().to(DEVICE) # L=16 -> Head=8
        H_tail_t = torch.tensor(y_target_q[:, 8:]).float().to(DEVICE)
        
        # --- Evaluate Baseline ---
        with torch.no_grad():
            p_b = baseline(X_t, H_head_t)
            mse_b = nn.MSELoss()(p_b, H_tail_t).item()
            task_mses_base.append(mse_b)
        
        # --- Evaluate Neural RTE ---
        with torch.no_grad():
            z_p, c_p = decomposer(X_t[:, :8], H_head_t)
            
            # Neural Re-Ranking (using first sample in batch to drive selection for stability, 
            # or average Z. Here we do row-wise but usually selection is per-task.
            # For simplicity in this loop, we pick best candidate based on the FIRST sample's head)
            # In a real setting you might vote across the batch.
            # Pass only the first element: c_p.cpu()[0]
            best_y_prev = select_best_candidate_neural(
                lib, extender, z_p.cpu().numpy()[0], c_p.cpu()[0], H_head_t[0], x_query[0], k=20
            )
            
            # Note: The 'best_y_prev' returned is a numpy array (L,). 
            # But we have a batch of X. We need the candidate function evaluated on ALL X in batch.
            # We must recover the function handle.
            # Let's cheat slightly and just re-retrieve the function index to apply it to the whole batch.
            _, indices = lib.nn_engine.kneighbors(z_p.cpu().numpy()[0].reshape(1,-1), n_neighbors=20)
            candidates = indices[0]
            
            # Re-run selection logic properly to get the function handle
            best_err = float('inf')
            best_func_handle = None
            
            c_in = c_p.cpu()[0].reshape(1,1).to(DEVICE)
            x_in_single = X_t[0:1] # (1, L)
            h_head_single = H_head_t[0]
            
            for c_idx in candidates:
                cand_f = lib.functions[c_idx]['func']
                y_prev_np = cand_f(x_query[0:1].flatten())
                y_prev_tens = torch.tensor(y_prev_np).float().to(DEVICE).unsqueeze(0)
                
                pred_tens = extender(x_in_single, y_prev_tens, c_in)
                err = nn.MSELoss()(pred_tens[0, :8], h_head_single).item()
                if err < best_err:
                    best_err = err
                    best_func_handle = cand_f
            
            # Now apply best function to WHOLE batch
            y_prev_batch = best_func_handle(x_query)
            y_prev_batch_t = torch.tensor(y_prev_batch).float().to(DEVICE)
            
            # Final Extender Pass
            full_pred = extender(X_t, y_prev_batch_t, c_p) # c_p is (B, 1), we use it directly
            p_r = full_pred[:, 8:]
            
            mse_r = nn.MSELoss()(p_r, H_tail_t).item()
            task_mses_rte.append(mse_r)

    # Compute Stats
    def get_ci(data):
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(len(data))
        return mean, 1.96 * sem

    mean_b, ci_b = get_ci(task_mses_base)
    mean_r, ci_r = get_ci(task_mses_rte)
    
    print(f"Baseline MSE: {mean_b:.4f} ± {ci_b:.4f}")
    print(f"RTE MSE:      {mean_r:.4f} ± {ci_r:.4f}")
    
    visualize_results(outdir, lib, decomposer, extender, baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out_neural_length")
    args = parser.parse_args()
    run_experiment(args.outdir)