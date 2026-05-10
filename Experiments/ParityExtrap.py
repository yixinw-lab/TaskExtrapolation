import os
import gc
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

# ================= CONFIGURATION =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
D = 8                   
TEST_N = 6              
N_TEST_TASKS = 28       
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 32         

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

set_seed(SEED)

# ================= LIBRARY =================
class ParityTaskLibrary:
    def __init__(self, D):
        self.D = D
        self.tasks = []
        lengths = range(2, 6)
        for n in lengths:
            all_combs = list(combinations(range(D), n))
            for bits in all_combs:
                mask = 0
                for b in bits: mask |= (1 << b)
                self.tasks.append(mask)
        random.shuffle(self.tasks)

    def get_data(self, mask, n_samples):
        X = np.random.randint(0, 2, (n_samples, self.D))
        Y = []
        for row in X:
            val = 0
            for i in range(self.D):
                if (mask >> i) & 1: val ^= row[i]
            Y.append(val)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

    def get_parents(self, mask):
        parents = []
        for i in range(self.D):
            if (mask >> i) & 1:
                parent_mask = mask ^ (1 << i)
                if bin(parent_mask).count('1') >= 1:
                    parents.append((parent_mask, i))
        return parents

    def generate_all_masks(self, n):
        masks = []
        all_combs = list(combinations(range(self.D), n))
        for bits in all_combs:
            m = 0
            for b in bits: m |= (1 << b)
            masks.append(m)
        return masks

# ================= LLM WRAPPER =================
class QwenWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
        # STRICT GPU MAPPING: Prevents the model from spilling over into CPU/meta devices
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        # GRADIENT CHECKPOINTING: Saves significant VRAM during the backward pass
        base_model.gradient_checkpointing_enable()
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(base_model, config)

    def make_baseline_prompt(self, supp_x, supp_y, query_x, is_training=False):
        content = "Instruction: Compute the Output.\n###\n"
        for sx, sy in zip(supp_x, supp_y):
            content += f"In: {' '.join(map(str, sx))}\nOut: {int(sy)}\n\n"
        content += f"In: {' '.join(map(str, query_x))}\nOut:"
        messages = [{"role": "user", "content": content.strip()}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def make_extender_prompt(self, supp_x, supp_y, supp_a, supp_b, query_x, query_a, query_b, is_training=False):
        content = "Instruction: Compute the Output.\n###\n"
        for sx, sy, sa, sb in zip(supp_x, supp_y, supp_a, supp_b):
            content += f"In: {' '.join(map(str, sx))}\nAnchor: {sa}\nBit: {sb}\nOut: {int(sy)}\n\n"
        content += f"In: {' '.join(map(str, query_x))}\nAnchor: {query_a}\nBit: {query_b}\nOut:"
        messages = [{"role": "user", "content": content.strip()}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def train_step(self, prompt_list, target_strs):
        self.model.train()
        full_texts = [p + t for p, t in zip(prompt_list, target_strs)]

        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        labels = inputs.input_ids.clone()
        labels[inputs.attention_mask == 0] = -100

        # SAFE MASKING: Ensure only the final token (the "0" or "1") is learned.
        for i in range(len(full_texts)):
            seq_len = inputs.attention_mask[i].sum().item()
            labels[i, :seq_len-1] = -100

        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    def batch_validate_loss(self, prompt_list, target_strs):
        self.model.eval()
        all_losses = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        with torch.no_grad():
            for i in range(0, len(prompt_list), BATCH_SIZE):
                batch_prompts = prompt_list[i:i+BATCH_SIZE]
                batch_targets = target_strs[i:i+BATCH_SIZE]

                # Left-pad so the last token is always aligned at the end of the sequence for easy extraction
                self.tokenizer.padding_side = 'left' 
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                self.tokenizer.padding_side = 'right'

                outputs = self.model(**inputs)
                # Grab the logits for the very last token of the generated prompt
                next_token_logits = outputs.logits[:, -1, :].contiguous() 

                target_ids = []
                for t in batch_targets:
                    t_id = self.tokenizer.encode(t, add_special_tokens=False)[-1]
                    target_ids.append(t_id)

                target_tensor = torch.tensor(target_ids, dtype=torch.long).to(DEVICE)
                
                # Single-token Cross Entropy Loss
                loss = loss_fct(next_token_logits, target_tensor)
                all_losses.extend(loss.cpu().tolist())

        return all_losses

    def predict(self, prompt):
        self.model.eval()
        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        self.tokenizer.padding_side = 'right'
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=4, pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False, temperature=None, top_p=None
            )
            
        input_len = inputs.input_ids.shape[1]
        raw_pred = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        
        extracted = "X"
        for char in raw_pred:
            if char in ["0", "1"]:
                extracted = char
                break
        return extracted

# ================= EXPERIMENT PHASES =================

def run_baseline(lib, eval_datasets):
    print("\n\n" + "="*40)
    print(" PHASE 1: DIRECT BASELINE (Few-Shot)")
    print("="*40)
    
    wrapper = QwenWrapper(MODEL_NAME)
    opt = optim.AdamW(wrapper.model.parameters(), lr=2e-4)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=50, num_training_steps=600)
    grad_accum_steps = 16
    
    print(f"\n--- Training Baseline (600 Steps, Effective Batch {grad_accum_steps}) ---")
    for step in tqdm(range(600)):
        for _ in range(grad_accum_steps):
            t_idx = np.random.randint(len(lib.tasks))
            mask = lib.tasks[t_idx]
            
            x, y = lib.get_data(mask, 6) 
            supp_x = x[:5].numpy().astype(int)
            supp_y = y[:5].numpy()
            query_x = x[5].numpy().astype(int)
            query_y = y[5].numpy()
            
            p = wrapper.make_baseline_prompt(supp_x, supp_y, query_x, is_training=True)
            loss = wrapper.train_step([p], [str(int(query_y))])
            loss = loss / grad_accum_steps
            loss.backward()
            
        opt.step()
        scheduler.step()
        opt.zero_grad()

    print("\n--- Evaluating Baseline ---")
    acc_base = 0; total = 0
    
    for t_idx, data in enumerate(eval_datasets):
        correct_b = 0
        supp_x, supp_y, query_x, query_y = data['supp_x'], data['supp_y'], data['query_x'], data['query_y']
        
        for i in range(len(query_x)):
            prompt = wrapper.make_baseline_prompt(supp_x, supp_y, query_x[i], is_training=False)
            pred = wrapper.predict(prompt)
            if pred == str(int(query_y[i])): correct_b += 1
            
        print(f"Task {t_idx+1}/{N_TEST_TASKS}: Baseline Correct = {correct_b}/{len(query_x)}")
        acc_base += correct_b
        total += len(query_x)
        
    print(f"\n>>> FINAL BASELINE ACCURACY: {acc_base/total:.2%}")
    
    # AGGRESSIVE MEMORY CLEANUP
    del wrapper, opt, scheduler
    clean_memory()
    return acc_base / total


def run_extender(lib, eval_datasets, test_masks):
    print("\n\n" + "="*40)
    print(" PHASE 2: COMPOSITIONAL EXTENDER")
    print("="*40)
    
    wrapper = QwenWrapper(MODEL_NAME)
    opt = optim.AdamW(wrapper.model.parameters(), lr=2e-4)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=50, num_training_steps=600)
    grad_accum_steps = 16
    
    print(f"\n--- Training Extender (600 Steps, Effective Batch {grad_accum_steps}) ---")
    for step in tqdm(range(600)):
        for _ in range(grad_accum_steps):
            parents = []
            while not parents:
                t_idx = np.random.randint(len(lib.tasks))
                mask = lib.tasks[t_idx]
                parents = lib.get_parents(mask)
            
            x, y = lib.get_data(mask, 6) 
            p_mask, bit_idx = random.choice(parents)
            
            # Compute anchors and bits for the 6 samples
            anchors = []
            bits = []
            for i in range(6):
                row = x[i].numpy().astype(int)
                p_val = 0
                for b in range(D):
                    if (p_mask >> b) & 1: p_val ^= row[b]
                anchors.append(p_val)
                bits.append(row[bit_idx])
                
            supp_x = x[:5].numpy().astype(int); supp_y = y[:5].numpy()
            query_x = x[5].numpy().astype(int); query_y = y[5].numpy()
            
            p = wrapper.make_extender_prompt(
                supp_x, supp_y, anchors[:5], bits[:5], 
                query_x, anchors[5], bits[5], is_training=True
            )
            
            loss = wrapper.train_step([p], [str(int(query_y))])
            loss = loss / grad_accum_steps
            loss.backward()
            
        opt.step()
        scheduler.step()
        opt.zero_grad()

    print("\n--- Evaluating Extender (Leave-One-Out Support Search) ---")
    candidates = lib.generate_all_masks(TEST_N - 1)
    acc_reason = 0; acc_oracle = 0; total = 0
    
    for t_idx, data in enumerate(eval_datasets):
        supp_x, supp_y, query_x, query_y = data['supp_x'], data['supp_y'], data['query_x'], data['query_y']
        t_mask = test_masks[t_idx]
        
        # 1. Search Logic using Leave-One-Out Few-Shot Context
        cand_prompts_flat = []
        cand_targets_flat = []
        cand_meta = [] 
        
        for c_mask in candidates:
            for bit in range(D):
                # Calculate properties for all 5 support points under this candidate
                s_anchors = []
                s_bits = []
                for i in range(len(supp_x)):
                    row = supp_x[i]
                    p_val = 0
                    for b in range(D):
                        if (c_mask >> b) & 1: p_val ^= row[b]
                    s_anchors.append(p_val)
                    s_bits.append(row[bit])
                
                # Build Leave-One-Out Prompts
                for i in range(len(supp_x)):
                    loo_supp_x = np.delete(supp_x, i, axis=0)
                    loo_supp_y = np.delete(supp_y, i, axis=0)
                    loo_anchors = s_anchors[:i] + s_anchors[i+1:]
                    loo_bits = s_bits[:i] + s_bits[i+1:]
                    
                    p = wrapper.make_extender_prompt(
                        loo_supp_x, loo_supp_y, loo_anchors, loo_bits,
                        supp_x[i], s_anchors[i], s_bits[i], is_training=False
                    )
                    cand_prompts_flat.append(p)
                    cand_targets_flat.append(str(int(supp_y[i])))
                    
                cand_meta.append((c_mask, bit))
                
        flat_losses = wrapper.batch_validate_loss(cand_prompts_flat, cand_targets_flat)
        S_LEN = len(supp_x)
        cand_losses = [sum(flat_losses[i*S_LEN:(i+1)*S_LEN]) for i in range(len(cand_meta))]
        
        best_idx = np.argmin(cand_losses)
        r_mask, r_bit = cand_meta[best_idx]
        
        # 2. Oracle Data setup
        o_parents = []
        for i in range(D):
            if (t_mask >> i) & 1:
                pm = t_mask ^ (1 << i)
                if bin(pm).count('1') >= 1: o_parents.append((pm, i))
        o_mask, o_bit = o_parents[0] if o_parents else (0,0)
        
        print(f"\n[DEBUG] Task {t_idx+1} Target Mask: {bin(t_mask)}")
        print(f"[DEBUG] Search Retrieved -> Parent: {bin(r_mask)}, Bit: {r_bit} (Loss: {cand_losses[best_idx]:.4f})")
        if (r_mask, r_bit) in o_parents:
            print("[DEBUG] SUCCESS: Identified a valid ground truth decomposition.")
        else:
            print("[DEBUG] FAILED: Search picked a spurious decomposition.")

        # 3. Final Query Predictions (Context = All 5 Support Examples)
        correct_r = 0; correct_o = 0
        for i in range(len(query_x)):
            row = query_x[i]
            true_y = str(int(query_y[i]))
            
            # Reason Pred
            pv = 0
            for b in range(D):
                if (r_mask >> b) & 1: pv ^= row[b]
            
            # We need the support anchors/bits evaluated under the Reason mask
            r_supp_a = []; r_supp_b = []
            for s_row in supp_x:
                sav = 0
                for b in range(D):
                    if (r_mask >> b) & 1: sav ^= s_row[b]
                r_supp_a.append(sav); r_supp_b.append(s_row[r_bit])
                
            pred_r = wrapper.predict(wrapper.make_extender_prompt(
                supp_x, supp_y, r_supp_a, r_supp_b, row, pv, row[r_bit], is_training=False
            ))
            if pred_r == true_y: correct_r += 1
            
            # Oracle Pred
            opv = 0
            for b in range(D):
                if (o_mask >> b) & 1: opv ^= row[b]
                
            o_supp_a = []; o_supp_b = []
            for s_row in supp_x:
                sav = 0
                for b in range(D):
                    if (o_mask >> b) & 1: sav ^= s_row[b]
                o_supp_a.append(sav); o_supp_b.append(s_row[o_bit])
                
            pred_o = wrapper.predict(wrapper.make_extender_prompt(
                supp_x, supp_y, o_supp_a, o_supp_b, row, opv, row[o_bit], is_training=False
            ))
            if pred_o == true_y: correct_o += 1
            
        print(f"--> Task {t_idx+1} Score: Reason={correct_r}/10 | Oracle={correct_o}/10")
        acc_reason += correct_r
        acc_oracle += correct_o
        total += len(query_x)

    print(f"\n>>> FINAL EXTENDER RESULTS: Reason {acc_reason/total:.2%} | Oracle {acc_oracle/total:.2%}")
    
    # AGGRESSIVE MEMORY CLEANUP
    del wrapper, opt, scheduler
    clean_memory()
    return acc_reason / total, acc_oracle / total

def main():
    clean_memory()
    lib = ParityTaskLibrary(D)
    
    print(f"\nGenerating Evaluation Datasets (Testing all {N_TEST_TASKS} exhaustive size-{TEST_N} combinations)...")
    test_masks = lib.generate_all_masks(TEST_N)
    
    eval_datasets = []
    for t_mask in test_masks:
        X_all = np.random.randint(0, 2, (15, D))
        Y_all = []
        for row in X_all:
            val = 0
            for i in range(D):
                if (t_mask >> i) & 1: val ^= row[i]
            Y_all.append(val)
            
        eval_datasets.append({
            'supp_x': X_all[:5],  
            'supp_y': Y_all[:5],
            'query_x': X_all[5:15], 
            'query_y': Y_all[5:15]
        })

    base_acc = run_baseline(lib, eval_datasets)
    reason_acc, oracle_acc = run_extender(lib, eval_datasets, test_masks)
    
    print("\n" + "="*40)
    print("🏆 SUMMARY OF RESULTS 🏆")
    print("="*40)
    print(f"Baseline (Direct Few-Shot):  {base_acc:.2%}")
    print(f"Extender (Search + Reason):  {reason_acc:.2%}")
    print(f"Oracle (Ground Truth Sub):   {oracle_acc:.2%}")

if __name__ == "__main__":
    main()
