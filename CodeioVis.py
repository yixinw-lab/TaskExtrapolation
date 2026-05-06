import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

# Set non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 1. ATOMIC FUNCTIONS (Task Definitions)
# ==========================================
def deterministic_shuffle(s):
    L = len(s)
    if L == 0: return s
    multiplier = 3
    while True:
        a, b = multiplier, L
        while b: a, b = b, a % b
        if a == 1: break
        multiplier += 2
    return ''.join(s[(i * multiplier) % L] for i in range(L))

def repeat_str(s): return s * 3
def remove_vowels(s): return ''.join(ch for ch in s if ch not in 'aeiouAEIOU')
def remove_consonants(s): return ''.join(ch for ch in s if ch in 'aeiouAEIOU ') 
def sort_chars(s): return ''.join(sorted(s))
def sort_descending(s): return ''.join(sorted(s, reverse=True)) 
def reverse_words(s): return ' '.join(reversed(s.split()))
def add_prefix(s): return "PRE_" + s
def add_suffix(s): return s + "_SUF"
def interlace_str(s): 
    mid = len(s) // 2
    s1, s2 = s[:mid], s[mid:]
    result = []
    len1, len2 = len(s1), len(s2)
    for i in range(max(len1, len2)):
        if i < len1: result.append(s1[i])
        if i < len2: result.append(s2[i])
    return ''.join(result)
def rotate_str(s): return s[5:] + s[:5] if len(s) > 5 else s
def mirror_str(s): return s + s[::-1]
def alternate_case(s): return ''.join(ch.lower() if i % 2 == 0 else ch.upper() for i, ch in enumerate(s))
def shift_chars(s):
    def shift_char(ch):
        if 'a' <= ch <= 'z': return chr((ord(ch) - ord('a') + 1) % 26 + ord('a'))
        elif 'A' <= ch <= 'Z': return chr((ord(ch) - ord('A') + 1) % 26 + ord('A'))
        return ch
    return ''.join(shift_char(ch) for ch in s)
def vowel_to_number(s):
    mapping = {'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5', 'A': '1', 'E': '2', 'I': '3', 'O': '4', 'U': '5'}
    return ''.join(mapping.get(ch, ch) for ch in s)
def insert_separator(s): return '-'.join(s)
def duplicate_every_char(s): return ''.join(ch * 2 for ch in s)
def fancy_brackets(s): return ''.join("«" + ch + "»" for ch in s)
def compress_repeats(s):
    if not s: return s
    result = [s[0]]
    for ch in s[1:]:
        if ch != result[-1]: result.append(ch)
    return ''.join(result)
def recursive_reverse(s):
    if s == "": return s
    return recursive_reverse(s[1:]) + s[0]
def loop_concat(s): return s * 3
def while_rotate(s):
    count = 0
    n = 4
    while count < n and s:
        s = s[1:] + s[0]
        count += 1
    return s
def recursive_interlace(s):
    mid = len(s)//2
    s1, s2 = s[:mid], s[mid:]
    def recurse(a, b):
        if not a or not b: return a + b
        return a[0] + b[0] + recurse(a[1:], b[1:])
    return recurse(s1, s2)
def loop_filter_nonalpha(s): return ''.join(ch for ch in s if ch.isalpha())
def verify_even_length(s): return s if len(s) % 2 == 0 else s[:-1]
def backchain_add_digit(s):
    depth = 3
    def recurse(x, d):
        if d <= 0 or len(x) > 20: return x
        return recurse(x + "1", d - 1)
    return recurse(s, depth)
def backchain_palindrome(s):
    if s == s[::-1]: return s
    return s + s[::-1]
def identity_task(s): return s
def random_scramble(s):
    random.seed(abs(hash(s)) % (10**9)) 
    return ''.join(random.sample(s, len(s)))
def sort_then_reverse(s): return sort_chars(s)[::-1]
def reverse_then_upper(s): return s[::-1].upper()
def duplicate_manual(s):
    return interlace_str(s + s) 

atomic_functions = {
    'identity_task':        identity_task,
    'random_scramble':      random_scramble,
    'deterministic_shuffle': deterministic_shuffle,
    'repeat_str':           repeat_str,
    'remove_vowels':        remove_vowels,
    'remove_consonants':    remove_consonants,
    'sort_chars':           sort_chars,
    'sort_descending':      sort_descending,
    'reverse_words':        reverse_words,
    'add_prefix':           add_prefix,
    'add_suffix':           add_suffix,
    'interlace_str':        interlace_str,
    'rotate_str':           rotate_str,
    'mirror_str':           mirror_str,
    'alternate_case':       alternate_case,
    'shift_chars':          shift_chars,
    'vowel_to_number':      vowel_to_number,
    'insert_separator':     insert_separator,
    'duplicate_every_char': duplicate_every_char,
    'fancy_brackets':       fancy_brackets,
    'compress_repeats':     compress_repeats,
    'recursive_reverse':    recursive_reverse,
    'loop_concat':          loop_concat,
    'while_rotate':         while_rotate,
    'recursive_interlace':  recursive_interlace,
    'loop_filter_nonalpha': loop_filter_nonalpha,
    'verify_even_length':   verify_even_length,
    'backchain_add_digit':  backchain_add_digit,
    'backchain_palindrome': backchain_palindrome,
    'sort_then_reverse':    sort_then_reverse,
    'reverse_then_upper':   reverse_then_upper,
    'duplicate_manual':     duplicate_manual
}

# ==========================================
# 2. DATA UTILS (Space-Delimited Fix)
# ==========================================
def generate_io_pool(func, pool_size=1000):
    inputs = []
    for _ in range(pool_size):
        length = random.randint(5, 12) # Shorter to stay in context
        chars = 'abcdefghijklmnopqrstuvwxyz' 
        # Generate raw string
        raw_s = ''.join(random.choice(chars) for _ in range(length))
        inputs.append(raw_s)
    
    pairs = []
    for s in inputs:
        try:
            # 1. Apply logic to raw string first
            output_s = func(s)
            
            # 2. SPACE DELIMIT EVERYTHING
            # This forces GPT-2 to treat them as individual character tokens
            # preventing BPE artifacts from destroying the gradient signal.
            in_str = " ".join(list(s))
            out_str = " ".join(list(output_s))
            
            pairs.append((in_str, out_str))
        except:
            pass 
    return pairs

class FewShotTaskDataset(Dataset):
    def __init__(self, io_pool, tokenizer, num_shots=3, max_length=128, samples_per_epoch=200):
        self.io_pool = io_pool
        self.tokenizer = tokenizer
        self.num_shots = num_shots
        self.max_length = max_length
        self.samples_per_epoch = samples_per_epoch
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        examples = random.sample(self.io_pool, self.num_shots + 1)
        context_pairs = examples[:-1]
        target_input, target_output = examples[-1]
        
        prompt_text = ""
        for x, y in context_pairs:
            prompt_text += f"In: {x} Out: {y}\n"
        prompt_text += f"In: {target_input} Out:"
        
        full_text = prompt_text + f" {target_output}" + self.tokenizer.eos_token
        encoding = self.tokenizer(
            full_text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer(prompt_text)['input_ids'])
        
        if prompt_len < self.max_length:
            labels[:prompt_len] = -100
        labels[encoding['attention_mask'].flatten() == 0] = -100
        
        return {'input_ids': input_ids, 'attention_mask': encoding['attention_mask'].flatten(), 'labels': labels}

# ==========================================
# 3. LoRA WRAPPER
# ==========================================
class LoRAConv1DWrapper(nn.Module):
    def __init__(self, target_module, rank=8, alpha=16):
        super().__init__()
        self.target_module = target_module
        self.in_features = target_module.weight.shape[0]
        self.out_features = target_module.weight.shape[1]
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.target_module(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

# ==========================================
# 4. FISHER CALCULATION (Layer-Wise Norm Fix)
# ==========================================
class Task2Vec(nn.Module):
    def __init__(self, model, filter_term='lora'):
        super().__init__()
        self.model = model
        # Group parameters by layer to normalize them independently
        self.layers_params = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and filter_term in name.lower():
                # Extract layer index (e.g., "transformer.h.15.attn...")
                parts = name.split('.')
                layer_idx = -1
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                
                if layer_idx not in self.layers_params:
                    self.layers_params[layer_idx] = []
                self.layers_params[layer_idx].append(param)

    def compute_fisher_diagonal(self, dataloader, num_batches=20):
        self.model.eval() 
        device = next(self.model.parameters()).device
        
        # Initialize accumulators for each layer
        layer_fishers = {l: [] for l in self.layers_params}
        for l in layer_fishers:
            for p in self.layers_params[l]:
                layer_fishers[l].append(torch.zeros_like(p))
        
        count = 0
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = self.model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Accumulate squared gradients
            for l, params in self.layers_params.items():
                for idx, p in enumerate(params):
                    if p.grad is not None:
                        layer_fishers[l][idx] += (p.grad ** 2)
            
            self.model.zero_grad()
            count += 1
            
        # Normalize PER LAYER to balance "Loud" vs "Quiet" layers
        final_embedding = []
        sorted_layers = sorted(self.layers_params.keys())
        
        for l in sorted_layers:
            layer_vecs = []
            for idx, fisher_tensor in enumerate(layer_fishers[l]):
                if count > 0:
                    fisher_tensor /= count
                layer_vecs.append(fisher_tensor.flatten().detach().cpu())
            
            if not layer_vecs: continue
            
            # Concatenate params in this layer
            layer_flat = torch.cat(layer_vecs)
            
            # L2 NORMALIZE THE LAYER
            norm = torch.norm(layer_flat) + 1e-8
            layer_normalized = layer_flat / norm
            
            final_embedding.append(layer_normalized)
            
        if not final_embedding: return np.array([])
        return torch.cat(final_embedding).numpy()

# ==========================================
# 5. MAIN PIPELINE (Complete Fix)
# ==========================================
def main():
    print("Initialize Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    all_tasks = list(atomic_functions.keys())
    
    # ------------------------------------
    # PHASE 2: GLOBAL EMBEDDINGS (FINAL)
    # ------------------------------------
    print("\n" + "="*40)
    print("PHASE 2: EMBEDDING EXTRACTION")
    print("  - Strategy: Vanilla Probe")
    print("  - Tokenization: Space-Delimited")
    print("  - Layers: 10-32 (Logic Core)")
    print("="*40)
    
    embeddings = {}
    
    # SURGICAL LAYER SELECTION
    # Skip 0-9 (Syntax), Use 10-32 (Reasoning), Skip 33-35 (Output)
    target_layers = range(10, 32) 
    
    # INCREASED EPOCHS FOR CONVERGENCE
    PROBE_MAX_EPOCHS = 50 
    PROBE_TARGET_LOSS = 0.02

    for task_name in all_tasks:
        print(f"  Embedding: {task_name}...")
        
        # RELOAD VANILLA MODEL
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.to(device)
        
        # Inject LoRA
        lora_params = []
        for i in target_layers:
            block = model.transformer.h[i]
            block.attn.c_attn = LoRAConv1DWrapper(block.attn.c_attn, rank=8)
            lora_params.append(block.attn.c_attn.lora_A)
            lora_params.append(block.attn.c_attn.lora_B)
        
        model.to(device)
        
        # Freeze non-LoRA
        for name, param in model.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
        
        # Train Probe
        optim = torch.optim.AdamW(lora_params, lr=1e-3)
        pool = generate_io_pool(atomic_functions[task_name], pool_size=200)
        ds = FewShotTaskDataset(pool, tokenizer, num_shots=3)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        
        model.train()
        for epoch in range(PROBE_MAX_EPOCHS):
            total_loss = 0
            steps = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                optim.step()
                optim.zero_grad()
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss/steps
            if avg_loss < PROBE_TARGET_LOSS: 
                break

        # Compute Fisher
        task2vec = Task2Vec(model, filter_term='lora')
        fisher_loader = DataLoader(ds, batch_size=1)
        
        embeddings[task_name] = task2vec.compute_fisher_diagonal(fisher_loader, num_batches=30)
        
        del model
        torch.cuda.empty_cache()

    # ------------------------------------
    # PHASE 3: VISUALIZATION (FINAL)
    # ------------------------------------
    print("\n>>> Generating Visualizations (Normalized + Scramble Centered)...")
    names = list(embeddings.keys())
    
    task_groups = {
        'Sorting': ['sort_chars', 'sort_descending', 'sort_then_reverse'],
        'Reversal': ['reverse_words', 'recursive_reverse', 'mirror_str', 'reverse_then_upper'],
        'Filtering': ['remove_vowels', 'remove_consonants', 'loop_filter_nonalpha', 'compress_repeats'],
        'Cipher/Shift': ['shift_chars', 'vowel_to_number', 'alternate_case', 'rotate_str', 'while_rotate'],
        'Structure': ['interlace_str', 'recursive_interlace', 'duplicate_every_char', 'duplicate_manual', 'repeat_str', 'loop_concat'],
        'Formatting': ['add_prefix', 'add_suffix', 'insert_separator', 'fancy_brackets', 'verify_even_length'],
        'Logic': ['backchain_add_digit', 'backchain_palindrome', 'deterministic_shuffle', 'random_scramble', 'identity_task']
    }
    
    group_map = {}
    for task in names:
        found = False
        for group, members in task_groups.items():
            if task in members:
                group_map[task] = group
                found = True
                break
        if not found: group_map[task] = 'Other'
        
    unique_groups = list(set(group_map.values()))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_groups)))
    color_dict = dict(zip(unique_groups, colors))

    X_raw = np.array([embeddings[n] for n in names])

    # SCRAMBLE CENTERING
    baseline_task = 'random_scramble' 
    if baseline_task in names:
        print(f"  Centering embeddings relative to '{baseline_task}'...")
        base_idx = names.index(baseline_task)
        base_vec = X_raw[base_idx].copy()
        X_proc = X_raw - base_vec
    else:
        print(f"  WARNING: '{baseline_task}' not found. Using raw vectors.")
        X_proc = X_raw

    # L2 Norm for Direction
    norms = np.linalg.norm(X_proc, axis=1, keepdims=True)
    X = X_proc / (norms + 1e-8)

    # A. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(14, 10))
    for group in unique_groups:
        idxs = [i for i, n in enumerate(names) if group_map[n] == group]
        plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], label=group, color=color_dict[group], s=120, alpha=0.8, edgecolors='k')
        
    for i, n in enumerate(names):
        plt.annotate(n, (X_pca[i, 0], X_pca[i, 1]), fontsize=9, alpha=0.9, xytext=(3, 3), textcoords='offset points')
        
    plt.title(f"Task2Vec PCA (Layer-Normed + {baseline_task} Centered)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('global_pca_final.png')
    plt.close()
    
    # B. t-SNE
    print("  Running t-SNE...")
    perp = min(len(names) // 2, 15)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(14, 10))
    for group in unique_groups:
        idxs = [i for i, n in enumerate(names) if group_map[n] == group]
        plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], label=group, color=color_dict[group], s=120, alpha=0.8, edgecolors='k')
        
    for i, n in enumerate(names):
        plt.annotate(n, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=9, alpha=0.9, xytext=(3, 3), textcoords='offset points')
        
    plt.title(f"Task2Vec t-SNE (Layer-Normed + {baseline_task} Centered)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('global_tsne_final.png')
    plt.close()

    # C. Heatmap
    sim_matrix = np.dot(X, X.T)
    plt.figure(figsize=(16, 14))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=90, fontsize=9)
    plt.yticks(range(len(names)), names, fontsize=9)
    plt.title(f"Cosine Similarity (Layer-Normed + {baseline_task} Centered)")
    plt.tight_layout()
    plt.savefig('global_heatmap_final.png')
    plt.close()

    print("\nDone! Check the '_final.png' files.")

if __name__ == "__main__":
    main()
