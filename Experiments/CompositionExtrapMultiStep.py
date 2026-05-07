import argparse
import os
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# --- Global Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPOSITION_DEPTH = 3
SEARCH_K = 60 # Increased to cast a wider net for the parallel decomposer

# --- OOD Train/Test Split (Systematic Compositional Generalization) ---
FAMILIES = ['poly', 'sin', 'tanh']
ALL_COMBOS = list(itertools.product(FAMILIES, repeat=COMPOSITION_DEPTH))

TRAIN_COMBOS = []
TEST_COMBOS = []

# SYSTEMATIC HOLD-OUT: We hold out any composition where 'tanh' is applied directly to 'sin'.
for combo in ALL_COMBOS:
    has_tanh_of_sin = False
    for i in range(len(combo) - 1):
        if combo[i] == 'sin' and combo[i+1] == 'tanh':
            has_tanh_of_sin = True
            break
    if has_tanh_of_sin:
        TEST_COMBOS.append(combo)
    else:
        TRAIN_COMBOS.append(combo)

print(f"Total Combinations: {len(ALL_COMBOS)}")
print(f"Training Combinations (In-Distribution): {len(TRAIN_COMBOS)}")
print(f"Testing Combinations (Systematic OOD Hold-Out): {len(TEST_COMBOS)}")

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
        self.input_domain = torch.tensor(
            input_domain).float().reshape(-1, 1).to(DEVICE)
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
            y_true = torch.from_numpy(
                library.dense_curves[idx]).float().reshape(-1, 1).to(DEVICE)
            opt.zero_grad()
            y_pred = self.probe(self.input_domain)
            loss_fn(y_pred, y_true).backward()
            opt.step()

    def _get_raw_gradients(self, func_idx, library):
        self.probe.zero_grad()
        y_true = torch.from_numpy(
            library.dense_curves[func_idx]).float().reshape(-1, 1).to(DEVICE)
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
        raw_grads = [self._get_raw_gradients(
            i, library) for i in range(len(library.functions))]
        raw_grads = np.stack(raw_grads)
        print("Fitting PCA...")
        self.embeddings = self.pca.fit_transform(raw_grads)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        return self.embeddings

# --- 1. The Primitive Library ---
class PrimitiveLibrary:
    # L bumped to 100 for higher resolution to avoid aliasing
    def __init__(self, n_per_family=1000, L=100, seed=42):
        self.rng = np.random.default_rng(seed)
        self.L = L
        self.x_range = np.linspace(-5, 5, L).astype(np.float32)
        self.functions = []
        self.dense_curves = []
        self.family_indices = {'poly': [], 'sin': [], 'tanh': []}
        print(f"Populating Library ({n_per_family}/family)...")
        self._populate(n_per_family)
        self.dense_curves = np.stack(self.dense_curves)
        self.types_list = [f['type'] for f in self.functions]

    def _populate(self, n):
        idx_counter = 0

        # NORMALIZATION HELPER: Forces range to exactly [-5, 5] over the [-5, 5] domain
        def normalize(raw_f):
            y_vals = raw_f(self.x_range)
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            if y_max - y_min < 1e-6:
                scale = 1.0
                offset = -y_min
            else:
                scale = 10.0 / (y_max - y_min)
                offset = -5.0 - y_min * scale
            return lambda x, f=raw_f, s=scale, o=offset: f(x) * s + o

        # Poly
        for _ in range(n):
            a, b, c = self.rng.uniform(-1.0, 1.0), self.rng.uniform(-1.0, 1.0), self.rng.uniform(-1.0, 1.0)
            raw_f = (lambda a, b, c: lambda x: a*x**2 + b*x + c)(a, b, c)
            f = normalize(raw_f)
            
            self.functions.append({'func': f, 'type': 'poly'})
            self.dense_curves.append(f(self.x_range))
            self.family_indices['poly'].append(idx_counter)
            idx_counter += 1

        # Sin
        for _ in range(n):
            w, phi = self.rng.uniform(0.5, 1.5), self.rng.uniform(0, np.pi)
            raw_f = (lambda w, phi: lambda x: np.sin(w*x + phi))(w, phi)
            f = normalize(raw_f)
            
            self.functions.append({'func': f, 'type': 'sin'})
            self.dense_curves.append(f(self.x_range))
            self.family_indices['sin'].append(idx_counter)
            idx_counter += 1

        # Tanh
        for _ in range(n):
            w, c = self.rng.uniform(0.5, 2.0), self.rng.uniform(-2, 2)
            raw_f = (lambda w, c: lambda x: np.tanh(w*(x - c)))(w, c)
            f = normalize(raw_f)
            
            self.functions.append({'func': f, 'type': 'tanh'})
            self.dense_curves.append(f(self.x_range))
            self.family_indices['tanh'].append(idx_counter)
            idx_counter += 1

    def compute_task2vec_embeddings(self):
        self.encoder = Task2VecEncoder(self.x_range, z_dim=16)
        self.embeddings = self.encoder.fit_transform_library(self)
        self.nn_engine = NearestNeighbors(metric='euclidean')
        self.nn_engine.fit(self.embeddings)

    def retrieve_top_k(self, z_query, k=10):
        _, indices = self.nn_engine.kneighbors(
            z_query.reshape(1, -1), n_neighbors=k)
        indices = indices[0]
        curves = self.dense_curves[indices]
        return curves

# --- 2. Models ---

class ParallelDecomposer(nn.Module):
    def __init__(self, L=100, z_dim=16, hidden_dim=128):
        super().__init__()
        in_dim = (L // 2) * 2
        # FFN replaces the RNN entirely to avoid autoregressive traps
        self.encoder = mlp([in_dim, 256, 256, hidden_dim])
        self.z_head = nn.Linear(hidden_dim, z_dim * COMPOSITION_DEPTH)

    def forward(self, x, h, depth=COMPOSITION_DEPTH):
        sort_idx = torch.argsort(x, dim=1)
        x_sorted = torch.gather(x, 1, sort_idx)
        h_sorted = torch.gather(h, 1, sort_idx)
        fin = torch.cat([x_sorted, h_sorted], dim=-1)
        
        hidden = self.encoder(fin)
        z_flat = self.z_head(hidden) # Predicts all embeddings simultaneously
        
        z_preds = []
        for d in range(depth):
            z_preds.append(z_flat[:, d*16 : (d+1)*16])
        return z_preds

class RecursiveComposer(nn.Module):
    def __init__(self, L=100):
        super().__init__()
        input_dim = L + L
        self.block1 = mlp([input_dim, 256, 256, 128])
        self.block2 = mlp([128, 128, 64])
        self.out = mlp([64, 64, L])

    def forward(self, current_state, primitive_desc):
        fin = torch.cat([current_state, primitive_desc], dim=-1)
        return self.out(self.block2(self.block1(fin)))

class InductiveBaseline(nn.Module):
    def __init__(self, L=100):
        super().__init__()
        input_dim = L + (L // 2)
        self.block1 = mlp([input_dim, 256, 256, 128])
        self.block2 = mlp([128, 128, 64])
        self.out = mlp([64, 64, L // 2])

    def forward(self, x, h_head):
        fin = torch.cat([x, h_head], dim=-1)
        return self.out(self.block2(self.block1(fin))).squeeze(-1)

# --- 3. Data Generation ---
def generate_batch_multistep(library, batch_size, depth=COMPOSITION_DEPTH, mode='train'):
    L = library.L
    x_query = library.rng.uniform(-5.0, 5.0,
                                  size=(batch_size, L)).astype(np.float32)
    batch_X = []
    batch_states = [[] for _ in range(depth + 1)]
    batch_Y_list = [[] for _ in range(depth)]
    batch_Z_list = [[] for _ in range(depth)]
    valid_combos = TRAIN_COMBOS if mode == 'train' else TEST_COMBOS

    for i in range(batch_size):
        recipe = random.choice(valid_combos)
        func_indices = [np.random.choice(
            library.family_indices[family_type]) for family_type in recipe]
        funcs = [library.functions[idx]['func'] for idx in func_indices]
        y_val = x_query[i]
        batch_states[0].append(y_val)

        for d in range(depth):
            y_val = funcs[d](y_val)
            batch_states[d + 1].append(y_val)
            batch_Y_list[d].append(library.dense_curves[func_indices[d]])
            batch_Z_list[d].append(library.embeddings[func_indices[d]])
        batch_X.append(x_query[i])

    X = torch.tensor(np.stack(batch_X)).float().to(DEVICE)
    states = [torch.tensor(np.stack(s)).float().to(DEVICE)
              for s in batch_states]
    Hfull = states[-1]
    Hh = Hfull[:, :L//2]
    Ht = Hfull[:, L//2:]
    Y_tensors = [torch.tensor(np.stack(batch_Y_list[d])).float().to(
        DEVICE) for d in range(depth)]
    Z_tensors = [torch.tensor(np.stack(batch_Z_list[d])).float().to(
        DEVICE) for d in range(depth)]

    return X, Y_tensors, Z_tensors, Hh, Ht, states

# --- 4. Logic & Re-Ranking ---
def select_best_sequence(library, z_preds_list, h_head_true, x_head_query, depth=COMPOSITION_DEPTH, k=SEARCH_K):
    candidates_curves = []
    for d in range(depth):
        # We only retrieve the curve DATA, not the functions
        curves_d = library.retrieve_top_k(z_preds_list[d], k=k)
        candidates_curves.append(curves_d)

    best_error = float('inf')
    best_seq_curves = [c[0] for c in candidates_curves]
    target = h_head_true.cpu().numpy().flatten()
    
    x_range = library.x_range

    for indices in itertools.product(range(k), repeat=depth):
        y_val = x_head_query
        for d in reversed(range(depth)):
            # INTERPOLATION TRICK: Evaluate the retrieved curve strictly using its data points!
            curve_data = candidates_curves[d][indices[d]]
            y_val = np.interp(y_val, x_range, curve_data)
            
        mse = np.mean((y_val - target)**2)
        if mse < best_error:
            best_error = mse
            best_seq_curves = [candidates_curves[d][indices[d]] for d in range(depth)]
            
    return best_seq_curves

def compare_models(library, decomp_rte, inductive_model, n_trials=1000):
    print(f"\n--- Comparing Inductive vs RTE ({n_trials} trials, Depth={COMPOSITION_DEPTH}) ---")
    print(f"--- Evaluating strictly on {len(TEST_COMBOS)} OOD held-out recipes ---")
    mse_rte, mse_ind = [], []
    decomp_rte.eval()
    inductive_model.eval()
    L = library.L
    x_range = library.x_range

    with torch.no_grad():
        for i in tqdm(range(n_trials), desc="Evaluating OOD Targets"):
            out = generate_batch_multistep(
                library, batch_size=1, depth=COMPOSITION_DEPTH, mode='test')
            X, _, _, H_head, H_tail, _ = out
            X_head = X[:, :L//2]

            # 1. Inductive Baseline
            pred_ind = inductive_model(X, H_head)
            mse_ind.append(nn.MSELoss()(pred_ind, H_tail).item())

            # 2. RTE
            z_preds = decomp_rte(X_head, H_head, depth=COMPOSITION_DEPTH)
            z_preds_np = [z.cpu().numpy()[0] for z in z_preds]
            
            best_seq_curves = select_best_sequence(
                library, z_preds_np, H_head[0], X_head.cpu().numpy()[0], depth=COMPOSITION_DEPTH, k=SEARCH_K)

            # Evaluate tail strictly using linear interpolation on the retrieved DATA
            y_val = X.cpu().numpy()[0]
            for d in reversed(range(COMPOSITION_DEPTH)):
                curve_data = best_seq_curves[d]
                y_val = np.interp(y_val, x_range, curve_data)
                
            mse_rte.append(np.mean((y_val[L//2:] - H_tail.cpu().numpy()[0])**2))

    # Calculate Statistics (Mean, Std Dev, and 95% CI)
    mean_ind = np.mean(mse_ind)
    std_ind = np.std(mse_ind, ddof=1)
    ci_ind = 1.96 * (std_ind / np.sqrt(n_trials))

    mean_rte = np.mean(mse_rte)
    std_rte = np.std(mse_rte, ddof=1)
    ci_rte = 1.96 * (std_rte / np.sqrt(n_trials))

    print("\n" + "=" * 75)
    print(f"{'Model':<20} | {'Mean MSE':<15} | {'95% CI':<15} | {'Std Dev':<15}")
    print("-" * 75)
    print(f"{'Inductive (Base)':<20} | {mean_ind:<15.4f} | +/- {ci_ind:<11.4f} | {std_ind:<15.4f}")
    print(f"{'RTE (Recursive)':<20} | {mean_rte:<15.4f} | +/- {ci_rte:<11.4f} | {std_rte:<15.4f}")
    print("=" * 75 + "\n")

def visualize_comparison(outdir, library, decomp_rte, inductive_model):
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 24, 'axes.labelsize': 20,
                         'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 16, 'figure.figsize': (24, 8)})
    decomp_rte.eval()
    inductive_model.eval()

    for eval_idx in range(1, 4):
        recipe = random.choice(TEST_COMBOS)
        func_indices = [np.random.choice(
            library.family_indices[t]) for t in recipe]
        funcs = [library.functions[idx] for idx in func_indices]
        L = library.L
        x_range = library.x_range

        X_test_np = library.rng.uniform(-5.0,
                                        5.0, size=(200, L)).astype(np.float32)
        y_h_test = X_test_np.copy()
        for f in funcs:
            y_h_test = f['func'](y_h_test)

        X_test = torch.tensor(X_test_np).float().to(DEVICE)
        H_head = torch.tensor(y_h_test[:, :L//2]).float().to(DEVICE)
        H_tail = y_h_test[:, L//2:]
        X_head_only = X_test[:, :L//2]

        ind_flat = inductive_model(
            X_test, H_head).detach().cpu().numpy().flatten()
        
        z_preds = decomp_rte(X_head_only, H_head, depth=COMPOSITION_DEPTH)
        z_preds_np = [z.detach().cpu().numpy() for z in z_preds]

        rte_preds = []
        best_vis_seq = None
        for i in range(200):
            x_hq = X_test_np[i, :L//2]
            z_seq = [z_p[i] for z_p in z_preds_np]
            
            best_seq_curves = select_best_sequence(
                library, z_seq, H_head[i], x_hq, depth=COMPOSITION_DEPTH, k=SEARCH_K)

            # Test-time Data Interpolation
            y_val = X_test_np[i:i+1]
            for d in reversed(range(COMPOSITION_DEPTH)):
                curve_data = best_seq_curves[d]
                y_val = np.interp(y_val, x_range, curve_data)

            rte_preds.append(y_val[:, L//2:])
            if i == 0:
                best_vis_seq = best_seq_curves

        rte_flat = np.concatenate(rte_preds).flatten()
        true_flat = H_tail.flatten()
        mse_ind = np.mean((ind_flat - true_flat)**2)
        mse_rte = np.mean((rte_flat - true_flat)**2)

        fig, axes = plt.subplots(1, 3)
        ax = axes[0]
        ax.plot(library.x_range,
                library.dense_curves[func_indices[0]], 'b-', lw=3, label='True Inner')
        ax.plot(library.x_range,
                best_vis_seq[-1], 'b--', lw=2, label='RTE Inner (Data)')
        title_str = " o ".join([f['type'] for f in reversed(funcs)])
        ax.set_title(
            f"Unseen Test Combo:\n{title_str}", fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        legend = ax.legend(loc='upper right', fontsize=16)
        plt.setp(legend.get_texts(), fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(X_test_np[:, L//2:].flatten(), ind_flat,
                   c='red', s=15, alpha=0.6, label='Pred')
        ax.scatter(X_test_np[:, L//2:].flatten(), true_flat,
                   c='black', s=5, alpha=0.2, label='True')
        ax.set_title(
            f"Naive Baseline\nMSE: {mse_ind:.4f}", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylim(-5, 5)

        ax = axes[2]
        ax.scatter(X_test_np[:, L//2:].flatten(), rte_flat,
                   c='blue', s=15, alpha=0.6, label='Pred')
        ax.scatter(X_test_np[:, L//2:].flatten(), true_flat,
                   c='black', s=5, alpha=0.2, label='True')
        ax.set_title(
            f"Recursive RTE \nMSE: {mse_rte:.4f}", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylim(-5, 5)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"compare_{eval_idx}.png"))
        plt.close()

def run_experiment(outdir):
    os.makedirs(outdir, exist_ok=True)
    lib = PrimitiveLibrary(n_per_family=500, L=100) # L increased to 100
    lib.compute_task2vec_embeddings()

    d_rte = ParallelDecomposer(L=100, z_dim=16).to(DEVICE)
    composer = RecursiveComposer(L=100).to(DEVICE) 
    inductive_model = InductiveBaseline(L=100).to(DEVICE)

    opt_rte = optim.Adam(d_rte.parameters(), lr=1e-3)
    opt_comp = optim.Adam(composer.parameters(), lr=1e-3)
    opt_ind = optim.Adam(inductive_model.parameters(), lr=1e-3)

    iters = 25000
    print(f"Training models for {iters} steps at DEPTH={COMPOSITION_DEPTH}...")

    for it in range(1, iters+1):
        out = generate_batch_multistep(
            lib, 64, depth=COMPOSITION_DEPTH, mode='train')
        X, Y_tensors, Z_tensors, Hh, Ht, states = out
        
        # Dynamic slice based on L to avoid tensor shape mismatch
        X_head = X[:, :lib.L//2]

        # 1. Train Decomposer
        opt_rte.zero_grad()
        z_preds = d_rte(X_head, Hh, depth=COMPOSITION_DEPTH)
        loss_rte_decomp = sum(nn.MSELoss()(
            z_preds[d], Z_tensors[COMPOSITION_DEPTH - 1 - d]) for d in range(COMPOSITION_DEPTH))
        loss_rte_decomp.backward()
        opt_rte.step()

        # 2. Train Composer
        opt_comp.zero_grad()
        loss_c = 0
        for d in range(COMPOSITION_DEPTH):
            pred_state = composer(states[d], Y_tensors[d])
            loss_c += nn.MSELoss()(pred_state, states[d+1])
        loss_c = loss_c / COMPOSITION_DEPTH
        loss_c.backward()
        opt_comp.step()

        # 3. Train Inductive Baseline
        opt_ind.zero_grad()
        loss_i = nn.MSELoss()(inductive_model(X, Hh), Ht)
        loss_i.backward()
        opt_ind.step()

        if it % 2000 == 0:
            print(
                f"Iter {it}: RTE_Decomp={loss_rte_decomp.item():.3f} | RTE_Comp={loss_c.item():.3f} | Inductive={loss_i.item():.3f}")

    # Pass only the needed models (Composer is ignored at test time to avoid covariate shift)
    compare_models(lib, d_rte, inductive_model)
    visualize_comparison(outdir, lib, d_rte, inductive_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="out_multistep_rte")
    args = parser.parse_args()
    run_experiment(args.outdir)
