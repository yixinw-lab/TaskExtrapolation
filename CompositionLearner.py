import os
import sys
# =========================================================
# 0. SETUP STORAGE
# =========================================================
ocean_path = "" 
if os.path.exists(ocean_path):
    cache_dir = os.path.join(ocean_path, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = os.path.join(ocean_path, "huggingface_cache_bnb")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
    os.environ["XDG_CACHE_HOME"] = cache_dir
    print(f"✅ Redirected all caches to: {cache_dir}")

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset
from tqdm import tqdm
import numpy as np
import random
import itertools
import gc
import json
from math import gcd

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # LLM Config
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    max_seq_len = 4096      
    lora_r = 32             
    batch_size = 2
    grad_accum_steps = 4
    epochs = 4              
    lr = 2e-4               

    # Experiment Config
    # NOTE: These are targets. prepare_data will auto-adjust if total pairs < target.
    n_train_pairs = 1000    
    n_test_pairs = 200
    
    # Shot Config
    n_atomic_shots = 6      
    n_target_shots_llm = 3  
    
    # Search Config
    search_batch_size = 8   

    # Paths
    storage_path = "outputs"
    seed = 42

cfg = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ==========================================
# 2. ATOMIC FUNCTIONS
# ==========================================
def deterministic_shuffle(s):
    L = len(s)
    if L == 0: return s
    multiplier = 3
    while gcd(multiplier, L) != 1: multiplier += 2
    return ''.join(s[(i * multiplier) % L] for i in range(L))

def repeat_str(s, n): return s * n
def remove_vowels(s): return ''.join(ch for ch in s if ch not in 'aeiouAEIOU')
def sort_chars(s): return ''.join(sorted(s))
def reverse_words(s): return ' '.join(reversed(s.split()))
def add_prefix(s, pre): return pre + s
def add_suffix(s, suf): return s + suf
def interlace_str(s1, s2):
    result = []
    len1, len2 = len(s1), len(s2)
    for i in range(max(len1, len2)):
        if i < len1: result.append(s1[i])
        if i < len2: result.append(s2[i])
    return ''.join(result)
def rotate_str(s, n):
    if not s: return s
    n = n % len(s)
    return s[n:] + s[:n]
def mirror_str(s): return s + s[::-1]
def alternate_case(s): return ''.join(ch.lower() if i % 2 == 0 else ch.upper() for i, ch in enumerate(s))
def shift_chars(s, shift):
    def shift_char(ch):
        if 'a' <= ch <= 'z': return chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
        elif 'A' <= ch <= 'Z': return chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
        return ch
    return ''.join(shift_char(ch) for ch in s)
def vowel_to_number(s):
    mapping = {'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5', 'A': '1', 'E': '2', 'I': '3', 'O': '4', 'U': '5'}
    return ''.join(mapping.get(ch, ch) for ch in s)
def insert_separator(s, sep): return sep.join(s)
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
def loop_concat(s, n): return s * n
def while_rotate(s, n):
    count = 0
    while count < n and s:
        s = s[1:] + s[0]
        count += 1
    return s
def recursive_interlace(s1, s2):
    if not s1 or not s2: return s1 + s2
    return s1[0] + s2[0] + recursive_interlace(s1[1:], s2[1:])
def loop_filter_nonalpha(s): return ''.join([ch for ch in s if ch.isalpha()])
def verify_even_length(s): return s if len(s) % 2 == 0 else s[:-1]
def backchain_add_digit(s, depth):
    def has_digit(t): return any(ch.isdigit() for ch in t)
    transformations = [lambda t: t + "1", lambda t: "2" + t, lambda t: t.replace("a", "3"), lambda t: t[::-1]]
    def helper(t, d):
        if has_digit(t): return t
        if d == 0: return None
        for trans in transformations:
            res = helper(trans(t), d - 1)
            if res is not None: return res 
        return None
    res = helper(s, depth)
    return res if res is not None else s 
def backchain_palindrome(s, depth):
    if s == s[::-1]: return s
    if depth <= 0: return s
    return backchain_palindrome(s + s[::-1], depth - 1)
def identity(s): return s

ATOMIC_FUNCS = {
    'identity': identity, 'shuffle': deterministic_shuffle, 'sort': sort_chars, 'reverse': recursive_reverse,
    'mirror': mirror_str, 'remove_vowels': remove_vowels, 'upper_case': lambda s: s.upper(), 'lower_case': lambda s: s.lower(),
    'duplicate_chars': duplicate_every_char, 'vowel_to_num': vowel_to_number, 'compress': compress_repeats,
    'filter_alpha': loop_filter_nonalpha, 'verify_even': verify_even_length, 'fancy_brackets': fancy_brackets,
    'reverse_words': reverse_words, 'alternate_case': alternate_case, 'repeat_2': lambda s: repeat_str(s, 2),
    'add_prefix_x': lambda s: add_prefix(s, "x_"), 'add_suffix_y': lambda s: add_suffix(s, "_y"),
    'rotate_1': lambda s: rotate_str(s, 1), 'rotate_3': lambda s: rotate_str(s, 3), 'shift_1': lambda s: shift_chars(s, 1),
    'shift_13': lambda s: shift_chars(s, 13), 'sep_dash': lambda s: insert_separator(s, "-"),
    'loop_concat_2': lambda s: loop_concat(s, 2), 'while_rotate_1': lambda s: while_rotate(s, 1),
    'force_digit': lambda s: backchain_add_digit(s, 3), 'force_palindrome': lambda s: backchain_palindrome(s, 2),
    'interlace_self_rev': lambda s: interlace_str(s, s[::-1]), 'recursive_interlace_rev': lambda s: recursive_interlace(s, s[::-1]),
    'concat_self': lambda s: s + s
}

# DIRECTLY DERIVE NAMES
FUNC_NAMES = sorted(list(ATOMIC_FUNCS.keys()))

# ==========================================
# 3. GLOBAL MASKING REGISTRY
# ==========================================
class FunctionMasker:
    def __init__(self, func_names):
        sorted_names = sorted(func_names)
        self.real_to_masked = {}
        for i, name in enumerate(sorted_names):
            self.real_to_masked[name] = f"Func_{i:02d}"
    
    def get_mask(self, real_name):
        return self.real_to_masked.get(real_name, "Unknown_Func")

MASKER = FunctionMasker(FUNC_NAMES)

# ==========================================
# 4. DYNAMIC CONTEXT GENERATION
# ==========================================
def generate_random_string():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choices(chars, k=random.randint(6, 14)))

_ATOMIC_CACHE = {}

def get_atomic_examples(func_name, forced_inputs=None, total_shots=3):
    """
    Generates Context examples.
    1. First, generates examples for the 'forced_inputs' (the current task inputs).
    2. Then, fills the rest with cached random examples to reach 'total_shots'.
    """
    examples = []
    f = ATOMIC_FUNCS.get(func_name, identity)
    
    # 1. Add Forced Inputs (Context Alignment)
    if forced_inputs:
        for inp in forced_inputs:
            try:
                out = f(inp)
                if out and len(out) > 0:
                    examples.append(f"In: {inp} Out: {out}")
            except: pass
            
    # 2. Fill with Cached/Random if needed
    needed = total_shots - len(examples)
    if needed > 0:
        # Check cache for fillers
        key = (func_name, needed)
        if key in _ATOMIC_CACHE:
            examples.extend(_ATOMIC_CACHE[key])
        else:
            # Generate new fillers
            fillers = []
            attempts = 0
            while len(fillers) < needed and attempts < 100:
                inp = generate_random_string()
                try:
                    out = f(inp)
                    if out and len(out) > 0: fillers.append(f"In: {inp} Out: {out}")
                except: pass
                attempts += 1
            _ATOMIC_CACHE[key] = fillers
            examples.extend(fillers)
            
    # Limit to requested shots
    final_exs = examples[:total_shots]
    return "\n".join(final_exs)

def create_prompt(inputs, outputs, query, context_funcs=None, is_training=False):
    """
    Main Prompt Creator.
    Forces the Reference Context to include transformations for the 'inputs' + 'query'.
    """
    full_content = ""
    
    # Collect all inputs that are relevant to this task to align context
    # For training/inference, we want context for the few-shot inputs AND the query input
    forced_inputs = inputs + [query] if query else inputs

    # 1. Context Section
    if context_funcs:
        valid_funcs = [f for f in context_funcs if f is not None]
        
        full_content += "You are provided with the following Reference Functions. Study their behavior:\n"
        
        for i, real_fname in enumerate(valid_funcs):
            masked_name = MASKER.get_mask(real_fname)
            if real_fname in ATOMIC_FUNCS:
                # Dynamic generation based on CURRENT task inputs
                exs = get_atomic_examples(real_fname, forced_inputs=forced_inputs, total_shots=cfg.n_atomic_shots)
                full_content += f"\n--- {masked_name} ---\n{exs}\n"
        
        full_content += "\nNow, use the Reference Functions above to solve the Target Task:\n"
    else:
        full_content += "Perform the string transformation pattern seen below:\n"

    # 2. Few-Shot Examples
    shots_str = "\n".join([f"In: {i} Out: {o}" for i, o in zip(inputs, outputs)])
    
    full_content += f"{shots_str}\n\n### Query:\nIn: {query} Out:"
    return [{"role": "user", "content": full_content}]

def create_scoring_prompt(current_input, all_support_inputs, context_funcs):
    """
    Scoring Prompt.
    Used during Search. We want the context to be as rich as Inference.
    So we provide context examples for ALL support inputs, but we only ask 
    the model to solve 'current_input'.
    """
    full_content = ""
    
    # We want context for all support items to simulate the full task environment
    forced_inputs = all_support_inputs 

    if context_funcs:
        valid_funcs = [f for f in context_funcs if f is not None]
        full_content += "You are provided with the following Reference Functions. Study their behavior:\n"
        for i, real_fname in enumerate(valid_funcs):
            masked_name = MASKER.get_mask(real_fname)
            if real_fname in ATOMIC_FUNCS:
                exs = get_atomic_examples(real_fname, forced_inputs=forced_inputs, total_shots=cfg.n_atomic_shots)
                full_content += f"\n--- {masked_name} ---\n{exs}\n"
        full_content += "\nNow, use the Reference Functions above to solve the Target Task:\n"
    
    full_content += f"In: {current_input} Out:"
    return [{"role": "user", "content": full_content}]

# ==========================================
# 5. DATA PREP & SEARCH
# ==========================================
def prepare_data():
    all_pairs = list(itertools.product(FUNC_NAMES, repeat=2))
    random.Random(cfg.seed).shuffle(all_pairs)

    total_pairs = len(all_pairs)
    print(f"--> Total Unique Pairs Available: {total_pairs}")

    # DYNAMIC SPLIT LOGIC
    # 1. Secure the Test Set first
    n_test = min(cfg.n_test_pairs, int(total_pairs * 0.2)) # Cap test at 20% if total is small
    if n_test < cfg.n_test_pairs:
        print(f"Warning: Requested {cfg.n_test_pairs} test pairs, but forced to reduce to {n_test}.")

    # 2. Give EVERYTHING ELSE to Training
    n_train = total_pairs - n_test
    
    print(f"--> Splitting into: {n_train} Train | {n_test} Test")

    # Slice strictly
    test_tasks = all_pairs[:n_test]
    train_tasks = all_pairs[n_test:] 

    # SEARCH SPACE
    train_set_set = set(train_tasks)
    search_space = []
    for f in FUNC_NAMES:
        search_space.append((f, None))
    for p in all_pairs:
        if p not in train_set_set:
            search_space.append(p)
    print(f"--> Constructed Search Space: {len(search_space)} candidates")

    train_flat, train_context, test_eval = [], [], []

    # --- GENERATE TRAINING DATA ---
    for pair in tqdm(train_tasks, desc="Gen Train"):
        fa, fb = ATOMIC_FUNCS[pair[0]], ATOMIC_FUNCS[pair[1]]
        samples = []
        while len(samples) < 15:
            s = generate_random_string()
            try: samples.append((s, fb(fa(s))))
            except: pass
        
        for i in range(3):
            in_shots = [x[0] for x in samples[i:i+cfg.n_target_shots_llm]]
            out_shots = [x[1] for x in samples[i:i+cfg.n_target_shots_llm]]
            q_in, q_out = samples[i+cfg.n_target_shots_llm]
            
            # Baseline
            train_flat.append({
                "messages": create_prompt(in_shots, out_shots, q_in, None, is_training=True), 
                "completion": " " + q_out
            })
            # Compositional (Context aligned to inputs)
            train_context.append({
                "messages": create_prompt(in_shots, out_shots, q_in, list(pair), is_training=True), 
                "completion": " " + q_out
            })

    # --- GENERATE TEST DATA ---
    for pair in tqdm(test_tasks, desc="Gen Test"):
        fa, fb = ATOMIC_FUNCS[pair[0]], ATOMIC_FUNCS[pair[1]]
        samples = []
        required = cfg.n_target_shots_llm + 5
        while len(samples) < required:
            s = generate_random_string()
            try: samples.append((s, fb(fa(s))))
            except: pass
        
        llm_supp = samples[0:cfg.n_target_shots_llm]
        query = samples[-1]

        test_eval.append({
            "llm_support": llm_supp,
            "query_in": query[0],
            "query_out": query[1],
            "pair": pair
        })

    return train_flat, train_context, test_eval, search_space

def compute_candidate_losses(model, tokenizer, candidates, item):
    """
    Scores candidates by checking how well they explain the support set.
    """
    support_pairs = item['llm_support'] # List of (in, out)
    all_inputs = [x[0] for x in support_pairs] # Extract all input strings for context alignment
    
    candidate_scores = []
    loss_fct = CrossEntropyLoss(reduction='none')
    
    for i in range(0, len(candidates), cfg.search_batch_size):
        batch_cands = candidates[i : i + cfg.search_batch_size]
        batch_losses = [0.0] * len(batch_cands)
        
        # Accumulate loss over ALL support shots
        for s_in, s_out in support_pairs:
            texts = []
            for cand in batch_cands:
                # Use Scoring Prompt: Context(Aligned to ALL inputs) + Query(s_in)
                msgs = create_scoring_prompt(s_in, all_inputs, list(cand))
                prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                full_text = prompt_text + " " + s_out + tokenizer.eos_token
                texts.append(full_text)
            
            # Tokenize batch
            encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len).to("cuda")
            input_ids = encodings.input_ids
            labels = input_ids.clone()
            
            # Mask out the prompt
            for idx, cand in enumerate(batch_cands):
                msgs = create_scoring_prompt(s_in, all_inputs, list(cand))
                p_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                p_len = len(tokenizer(p_text, add_special_tokens=False)['input_ids'])
                labels[idx, :p_len] = -100
                labels[idx, encodings.attention_mask[idx] == 0] = -100 

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                
                losses = loss_fct(flat_logits, flat_labels)
                losses = losses.view(shift_labels.shape)
                row_losses = losses.sum(dim=1).cpu().numpy()
                
                for b_idx in range(len(batch_cands)):
                    batch_losses[b_idx] += row_losses[b_idx]

        candidate_scores.extend(batch_losses)
    
    return candidate_scores

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def train_model(dataset, run_name):
    clean_memory()
    print(f"\n[{run_name}] Loading LLM...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name, max_seq_length=cfg.max_seq_len,
        dtype=None, load_in_4bit=True
    )
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing=True
    )
    
    tokenizer.padding_side = "right"
    
    def formatting_func(examples):
        # 1. Handle SINGLE example case
        if isinstance(examples["completion"], str):
            msgs = examples["messages"]
            compl = examples["completion"]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + compl + tokenizer.eos_token
            return [text]
        # 2. Handle BATCH case
        texts = []
        for msgs, compl in zip(examples["messages"], examples["completion"]):
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + compl + tokenizer.eos_token)
        return texts

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=HFDataset.from_list(dataset),
        formatting_func=formatting_func, max_seq_length=cfg.max_seq_len,
        args=SFTConfig(
            per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum_steps,
            warmup_steps=5, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=10, output_dir=os.path.join(cfg.storage_path, run_name),
            optim="adamw_8bit", seed=cfg.seed, report_to="none"
        )
    )
    trainer.train()
    return model, tokenizer

def evaluate(model, tokenizer, test_data, mode, search_space=None):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    
    correct, total = 0, 0
    
    print(f"\nStarting Evaluation: Mode = {mode}")
    
    if len(test_data) == 0:
        print("Warning: Test data is empty.")
        return 0.0

    for i, item in enumerate(tqdm(test_data)):
        sup_in = [x[0] for x in item['llm_support']]
        sup_out = [x[1] for x in item['llm_support']]
        
        ctx = None
        
        # --- STRATEGY SELECTION ---
        if mode == "Baseline":
            ctx = None # No context
            
        elif mode == "Oracle":
            ctx = list(item['pair']) # Provided Ground truth (masked)
            
        elif mode == "Search":
            # BRUTE FORCE SEARCH on the 3 support examples
            scores = compute_candidate_losses(model, tokenizer, search_space, item)
            best_idx = np.argmin(scores)
            ctx = list(search_space[best_idx])
            
            if i % 20 == 0:
                masked_found = [MASKER.get_mask(c) for c in ctx if c]
                masked_true = [MASKER.get_mask(c) for c in item['pair']]
                print(f"\n[Debug] True: {masked_true} | Found: {masked_found} | Loss: {scores[best_idx]:.4f}")

        # --- FINAL INFERENCE ---
        msgs = create_prompt(sup_in, sup_out, item['query_in'], ctx)
        inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        pred = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).split('\n')[0].strip()

        if pred == item['query_out'].strip(): correct += 1
        total += 1
        
        if i % 10 == 0:
            print(f"  Item {i}: Acc so far: {correct/total:.4f}")

    return correct / total if total > 0 else 0.0

def main():
    set_seed(cfg.seed)
    clean_memory()
    
    train_flat, train_context, test_eval, search_space = prepare_data()
    results = {}
    
    print("\n\n=== RUNNING BASELINE ===")
    model_base, tok_base = train_model(train_flat, "Baseline_Model")
    results['Baseline'] = evaluate(model_base, tok_base, test_eval, "Baseline")
    del model_base, tok_base
    clean_memory()
    
    print("\n\n=== TRAINING COMPOSITIONAL MODEL ===")
    model_comp, tok_comp = train_model(train_context, "Comp_Model")
    
    print("\n=== RUNNING ORACLE EVAL ===")
    results['Oracle'] = evaluate(model_comp, tok_comp, test_eval, "Oracle")
    
    print("\n=== RUNNING SEARCH EVAL ===")
    results['Search'] = evaluate(model_comp, tok_comp, test_eval, "Search", search_space=search_space)
    
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))
    with open("final_results.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()