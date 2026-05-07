import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. CONFIGURATION & DATA GENERATION ---

def projectile_y(x, v0, theta_deg, g=9.81):
    """
    Returns y for a given x based on physics formula.
    """
    theta = np.radians(theta_deg)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = x * np.tan(theta) - (g * x**2) / (2 * v0**2 * np.cos(theta)**2)
    return y

class ProjectileData:
    def __init__(self, n_tasks=100, v_range=(30, 60), theta_range=(30, 60)):
        # Training data is limited to velocities between 30 and 60
        self.v_vals = np.random.uniform(*v_range, n_tasks).astype(np.float32)
        self.theta_vals = np.random.uniform(*theta_range, n_tasks).astype(np.float32)
        
        # Keep track of training trajectories for plotting background lines
        self.train_viz_params = list(zip(self.v_vals[:40], self.theta_vals[:40])) 

    def sample_batch(self, batch_size=32):
        """Samples data for Inductive training (Standard Inputs -> Output)"""
        indices = np.random.randint(0, len(self.v_vals), batch_size)
        v = self.v_vals[indices]
        theta = self.theta_vals[indices]
        x = np.random.uniform(0, 200, batch_size).astype(np.float32)
        y = projectile_y(x, v, theta)
        return torch.tensor(x).unsqueeze(1), torch.tensor(v).unsqueeze(1), torch.tensor(theta).unsqueeze(1), torch.tensor(y).unsqueeze(1)

    def sample_pairs(self, batch_size=32):
        """Samples pairs for Transductive training (Anchor -> Difference -> Target)"""
        idx1 = np.random.randint(0, len(self.v_vals), batch_size) # Anchor
        idx2 = np.random.randint(0, len(self.v_vals), batch_size) # Target
        
        v1, t1 = self.v_vals[idx1], self.theta_vals[idx1]
        v2, t2 = self.v_vals[idx2], self.theta_vals[idx2]
        
        x = np.random.uniform(0, 200, batch_size).astype(np.float32)
        y_anchor = projectile_y(x, v1, t1)
        y_target = projectile_y(x, v2, t2)
        
        # The model learns to interpret these differences
        dv = v2 - v1
        dt = t2 - t1
        
        return (torch.tensor(x).unsqueeze(1),
                torch.tensor(y_anchor).unsqueeze(1),
                torch.tensor(dv).unsqueeze(1),
                torch.tensor(dt).unsqueeze(1),
                torch.tensor(y_target).unsqueeze(1))

# --- 2. MODELS ---

class InductiveModel(nn.Module):
    """
    Standard Neural Network.
    Tries to memorize the function: f(x, v, theta) -> y
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, v, theta):
        return self.net(torch.cat([x, v, theta], dim=1))

class TransductiveModel(nn.Module):
    """
    Relational Neural Network.
    Tries to learn the shift: f(x, anchor_y, delta_v, delta_theta) -> target_y
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, y_anc, dv, dt):
        return self.net(torch.cat([x, y_anc, dv, dt], dim=1))

# --- 3. TRAINING & PLOTTING ---

def train_and_plot_horizontal_compact():
    # Setup Data (Note: Train Velocity is ONLY 30-60)
    train_data = ProjectileData(n_tasks=200, v_range=(30, 60), theta_range=(40, 50)) 
    
    ind_model = InductiveModel()
    trans_model = TransductiveModel()
    
    opt_ind = optim.Adam(ind_model.parameters(), lr=0.005)
    opt_trans = optim.Adam(trans_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Training models...")
    for step in range(1000):
        # Train Inductive
        x, v, t, y = train_data.sample_batch(64)
        loss_ind = criterion(ind_model(x, v, t), y)
        opt_ind.zero_grad(); loss_ind.backward(); opt_ind.step()
        
        # Train Transductive
        x, y_anc, dv, dt, y_tgt = train_data.sample_pairs(64)
        loss_trans = criterion(trans_model(x, y_anc, dv, dt), y_tgt)
        opt_trans.zero_grad(); loss_trans.backward(); opt_trans.step()

    # --- EVALUATION CONFIG ---
    ind_model.eval(); trans_model.eval()
    
    # 1. Anchor (Inside training range)
    ANC_V, ANC_THETA = 60.0, 45.0
    
    # 2. Target (OUTSIDE training range)
    TGT_V, TGT_THETA = 65.0, 45.0
    
    DV = TGT_V - ANC_V
    DT = TGT_THETA - ANC_THETA
    
    # Plot Domain
    x_plot = np.linspace(0, 650, 200).astype(np.float32)
    x_tensor = torch.tensor(x_plot).unsqueeze(1)
    
    # Ground Truths (Physics)
    y_anc_gt = projectile_y(x_plot, ANC_V, ANC_THETA)
    y_tgt_gt = projectile_y(x_plot, TGT_V, TGT_THETA)
    
    # Predictions
    with torch.no_grad():
        # Inductive Prediction
        y_ind = ind_model(x_tensor, torch.full_like(x_tensor, TGT_V), torch.full_like(x_tensor, TGT_THETA)).numpy().flatten()
        
        # Transductive Prediction
        y_anc_input = torch.tensor(y_anc_gt).unsqueeze(1)
        y_trans = trans_model(x_tensor, y_anc_input, torch.full_like(x_tensor, DV), torch.full_like(x_tensor, DT)).numpy().flatten()

    # --- PLOTTING (HORIZONTAL & COMPACT) ---
    # CHANGED: Reduced figure height to 5 to reduce vertical whitespace
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150, constrained_layout=True)
    
    FONT_TITLE = 18
    FONT_AXIS_LBL = 14
    FONT_LEGEND = 12
    FONT_ANNOT = 14
    
    def setup_ax(ax, title):
        ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=15)
        # CHANGED: Reduced Y-limit max from 350 to 200 to zoom in on trajectories
        ax.set_ylim(-20, 200)
        ax.set_xlim(-20, 650)
        ax.axhline(0, color='k', lw=2)
        ax.axis('off') 
        
        # Draw Cannon
        ax.add_patch(Rectangle((-10, -5), 20, 10, color='black', zorder=10))
        ax.text(0, -35, "Input", ha='center', fontsize=FONT_AXIS_LBL)
        
        # --- FIXED CASTLE POSITION ---
        landing_x = 430.7 
        castle_w, castle_h = 40, 30
        ax.add_patch(Rectangle((landing_x - castle_w/2, 0), castle_w, castle_h, color='forestgreen', alpha=0.3))
        ax.text(landing_x, -35, "Target Task\n(Castle)", ha='center', color='forestgreen', fontweight='bold', fontsize=FONT_AXIS_LBL)

        # Plot All Training Trajectories
        for v_t, th_t in train_data.train_viz_params:
            yt = projectile_y(x_plot, v_t, th_t)
            ax.plot(x_plot, yt, color='gray', alpha=0.1, lw=1)

    # PANEL A: Inductive (LEFT)
    ax = axes[0]
    setup_ax(ax, "A. Inductive Model (Standard)")
    ax.plot(x_plot, y_ind, color='red', lw=3, ls='--', label='Model Prediction')
    ax.plot(x_plot, y_tgt_gt, color='forestgreen', lw=1, ls=':', alpha=0.5, label='Truth')
    ax.legend(loc='upper right', frameon=False, fontsize=FONT_LEGEND)

    # PANEL B: Transductive (RIGHT)
    ax = axes[1]
    setup_ax(ax, "B. Transductive Model (Relational)")
    
    ax.plot(x_plot, y_anc_gt, color='royalblue', lw=2.5, label='Anchor Trajectory')
    ax.plot(x_plot, y_trans, color='forestgreen', lw=3, ls='--', label='Prediction')
    ax.plot(x_plot, y_tgt_gt, color='black', lw=1.5, ls=':', alpha=0.8, label='Truth')
    
    # Arrow showing the shift
    p_anc = np.argmax(y_anc_gt)
    p_tgt = np.argmax(y_trans)
    
    arrow = FancyArrowPatch((x_plot[p_anc], y_anc_gt[p_anc]), (x_plot[p_tgt], y_trans[p_tgt]),
                            arrowstyle='->,head_width=10,head_length=10', color='purple', lw=2, zorder=20)
    ax.add_patch(arrow)
    
    # Text annotation
    text_x = (x_plot[p_anc] + x_plot[p_tgt]) / 2
    text_y = (y_anc_gt[p_anc] + y_trans[p_tgt]) / 2 + 15
    ax.text(text_x, text_y, " Learned\nShift", color='purple', fontsize=FONT_ANNOT, fontweight='bold')

    ax.legend(loc='upper right', frameon=False, fontsize=FONT_LEGEND)
    
    # Save as PDF
    plt.savefig('fig1_horizontal_compact.pdf')
    print("Saved fig1_horizontal_compact.pdf")

if __name__ == "__main__":
    train_and_plot_horizontal_compact()
