import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ==========================================
# 1. Unified Task2Vec / Embedding Logic
# ==========================================

def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)

class UniversalProbe(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 -> 128 -> 128 -> 1
        self.net = mlp([1, 128, 128, 1])
        
    def forward(self, x):
        return self.net(x)

class ManifoldVisualizer:
    def __init__(self, x_domain):
        self.x_domain = torch.tensor(x_domain).float().reshape(-1, 1).to(DEVICE)
        self.probe = UniversalProbe().to(DEVICE)
        
    def pretrain_probe(self, y_curves, iterations=3000, lr=1e-3, batch_size=32):
        print(f"   -> Pre-training probe on {len(y_curves)} functions (Batched)...")
        self.probe.train()
        opt = optim.Adam(self.probe.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        y_tensor = torch.tensor(np.stack(y_curves)).float().to(DEVICE)
        n_funcs = len(y_curves)
        
        for _ in range(iterations):
            indices = np.random.choice(n_funcs, size=batch_size, replace=True)
            y_batch = y_tensor[indices].reshape(batch_size, -1, 1)
            x_batch = self.x_domain.unsqueeze(0).expand(batch_size, -1, -1)
            
            opt.zero_grad()
            y_pred = self.probe(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            opt.step()

    def compute_embeddings(self, y_curves, method='pca', n_neighbors=15):
        print("   -> Computing Gradients...")
        self.probe.zero_grad()
        y_tensor = torch.tensor(np.stack(y_curves)).float().to(DEVICE)
        raw_grads = []
        loss_fn = nn.MSELoss()
        
        # Compute gradients sample-by-sample
        for i in range(len(y_curves)):
            self.probe.zero_grad()
            y_true = y_tensor[i].reshape(-1, 1)
            y_pred = self.probe(self.x_domain)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            
            grads = []
            for p in self.probe.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1).cpu().numpy())
                else:
                    grads.append(np.zeros(p.numel()))
            raw_grads.append(np.concatenate(grads))
            
        raw_grads = np.stack(raw_grads)
        
        print(f"   -> Projecting using {method.upper()}...")
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            embeddings = reducer.fit_transform(raw_grads)
            
        elif method == 'isomap':
            # Pre-reduce with PCA to 50 dims to remove noise before Isomap
            # This makes the manifold structure much cleaner.
            pca_pre = PCA(n_components=50) 
            grads_compressed = pca_pre.fit_transform(raw_grads)
            
            reducer = Isomap(n_neighbors=n_neighbors, n_components=2)
            embeddings = reducer.fit_transform(grads_compressed)
            
        return embeddings

# ==========================================
# 2. Experiment 1: Length (Polynomial Degree)
#    * FIXED: L2 Normalization *
# ==========================================

def run_length_experiment():
    print("\n[1/3] Generating Length (Polynomial) Manifold...")
    rng = np.random.default_rng(SEED)
    x_range = np.linspace(-5, 5, 20).astype(np.float32)
    y_curves, labels = [], []
    
    # Generate 50 samples per degree 1..9
    for d in range(1, 10):
        for _ in range(50):
            c_list = []
            for p in range(d + 1):
                val = rng.uniform(-1.0, 1.0)
                c_list.append(val)
                
            y = np.zeros_like(x_range)
            for p, val in enumerate(c_list):
                y += val * (x_range ** p)
            
            # --- THE FIX ---
            # Normalize to Unit Energy so the probe sees SHAPE, not MAGNITUDE.
            norm = np.linalg.norm(y) + 1e-8
            y = y / norm
            
            y_curves.append(y)
            labels.append(d)
            
    vis = ManifoldVisualizer(x_range)
    vis.pretrain_probe(y_curves, iterations=3000)
    
    # Use ISOMAP to unroll the "complexity" manifold
    embeddings = vis.compute_embeddings(y_curves, method='isomap', n_neighbors=15)
    return embeddings, labels

# ==========================================
# 3. Experiment 2: Composition Structure
# ==========================================

def run_composition_experiment():
    print("\n[2/3] Generating Composition Manifold...")
    rng = np.random.default_rng(SEED)
    x_range = np.linspace(-5, 5, 50).astype(np.float32)
    y_curves, labels = [], []
    
    # 1. Poly
    for _ in range(50):
        a, b, c = rng.uniform(-0.1, 0.1), rng.uniform(-0.5, 0.5), rng.uniform(-2, 2)
        y = a*x_range**2 + b*x_range + c
        y_curves.append(y) # No normalization needed here, distinct enough
        labels.append("Poly")

    # 2. Sin
    for _ in range(50):
        A, w, phi = rng.uniform(0.5, 2.0), rng.uniform(0.5, 1.5), rng.uniform(0, np.pi)
        y = A * np.sin(w*x_range + phi)
        y_curves.append(y)
        labels.append("Sin")
        
    # 3. Sin(Poly)
    for _ in range(50):
        a, b, c = rng.uniform(-0.1, 0.1), rng.uniform(-0.5, 0.5), rng.uniform(-2, 2)
        inner = a*x_range**2 + b*x_range + c
        A, w, phi = rng.uniform(0.5, 2.0), rng.uniform(0.5, 1.5), rng.uniform(0, np.pi)
        y = A * np.sin(w*inner + phi)
        y_curves.append(y)
        labels.append("Sin(Poly)")
        
    vis = ManifoldVisualizer(x_range)
    vis.pretrain_probe(y_curves, iterations=3000)
    
    # Use ISOMAP to separate the topological clusters
    embeddings = vis.compute_embeddings(y_curves, method='isomap', n_neighbors=20)
    return embeddings, labels

# ==========================================
# 4. Experiment 3: Parametric Shift
# ==========================================

def run_parametric_experiment():
    print("\n[3/3] Generating Parametric Shift Manifold...")
    rng = np.random.default_rng(SEED)
    x_range = np.linspace(0, 20, 20).astype(np.float32)
    y_curves, labels = [], []
    
    regions = {
        'F1_1 (Train A)': {'a': (0.5, 1.5)},
        'F1_2 (Train B)': {'a': (1.5, 2.5)},
        'F2 (Extrap)':    {'a': (2.5, 3.5)}
    }
    
    for r_name, bounds in regions.items():
        for _ in range(50):
            a = rng.uniform(*bounds['a'])
            b, c = rng.uniform(-2,2), rng.uniform(-2,2)
            y = a*(x_range**2) + b*x_range + c
            y_curves.append(y)
            labels.append(r_name)
            
    vis = ManifoldVisualizer(x_range)
    vis.pretrain_probe(y_curves, iterations=3000)
    
    # Use PCA because the shift in 'a' is a linear transformation
    embeddings = vis.compute_embeddings(y_curves, method='pca')
    return embeddings, labels

# ==========================================
# Main Plotting Routine
# ==========================================

def main():
    emb_len, lab_len = run_length_experiment()
    emb_comp, lab_comp = run_composition_experiment()
    emb_param, lab_param = run_parametric_experiment()
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plt.subplots_adjust(wspace=0.25)
    
    # Plot 1: Length (Isomap)
    ax = axes[0]
    ax.set_title("Length Extrapolation (Isomap)\n(Manifold Unrolling w/ Norm)", fontsize=12)
    sc = ax.scatter(emb_len[:, 0], emb_len[:, 1], c=lab_len, cmap='viridis', alpha=0.7, edgecolor='k', s=50)
    cbar = plt.colorbar(sc, ax=ax, label="Degree")
    cbar.set_ticks(np.arange(1, 10))
    ax.set_xlabel("Isomap Dim 1"); ax.set_ylabel("Isomap Dim 2"); ax.grid(True, alpha=0.3)
    
    # Plot 2: Composition (Isomap)
    ax = axes[1]
    ax.set_title("Compositional Structure (Isomap)\n(Topological Separation)", fontsize=12)
    unique_labels = list(set(lab_comp))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, label in enumerate(unique_labels):
        indices = [k for k, x in enumerate(lab_comp) if x == label]
        ax.scatter(emb_comp[indices, 0], emb_comp[indices, 1], label=label, 
                   alpha=0.7, edgecolors='k', s=50, color=colors[i % len(colors)])
    ax.legend(); ax.set_xlabel("Isomap Dim 1"); ax.set_ylabel("Isomap Dim 2"); ax.grid(True, alpha=0.3)

    # Plot 3: Parametric (PCA)
    ax = axes[2]
    ax.set_title("Parametric Extrapolation (PCA)\n(Linear Axis Discovery)", fontsize=12)
    unique_labels = ['F1_1 (Train A)', 'F1_2 (Train B)', 'F2 (Extrap)'] 
    colors = ['tab:blue', 'tab:purple', 'tab:red']
    for i, label in enumerate(unique_labels):
        indices = [k for k, x in enumerate(lab_param) if x == label]
        ax.scatter(emb_param[indices, 0], emb_param[indices, 1], label=label, 
                   alpha=0.7, edgecolors='k', s=50, color=colors[i % len(colors)])
    ax.legend(); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("TaskManifolds_Final.png", dpi=150)
    print("\nSaved final visualization to TaskManifolds_Final.png")

if __name__ == "__main__":
    main()
