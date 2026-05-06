import os
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# ================= CONFIGURATION =================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
D = 12                  
TEST_N = 7              # We search for parents of length N-1 = 6
N_TRAIN_TASKS = 3000     
N_TEST_TASKS = 10       
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 64         # Batch size for the brute force search

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ================= LIBRARY =================
class ParityTaskLibrary:
    def __init__(self, D, num_tasks=3000):
        self.D = D
        self.tasks = []
        # Generate tasks of length 2 to 6
        lengths = range(2, 7)
        for n in lengths:
            all_combs = list(combinations(range(D), n))
            for bits in all_combs:
                mask = 0
                for b in bits: mask |= (1 << b)
                self.tasks.append(mask)
        random.shuffle(self.tasks)
        if len(self.tasks) > num_tasks: self.tasks = self.tasks[:num_tasks]

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

# ================= EXTENDER (LLM) =================
class QwenExtender:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        # INCREASED CAPACITY: Rank 64, Target All Linear Layers
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=64, 
            lora_alpha=128, 
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(base_model, config)
        self.model.print_trainable_parameters()

    def make_prompt(self, x, p_out, y=None, is_training=False):
        # Structured Chat Format for Qwen-Instruct
        # We provide examples in the context, but here we format a single sample 
        # as a clear instruction.
        
        # To simplify: We feed the raw string into the chat template
        # "user": Context... Question...
        # "assistant": Answer
        
        content = "You are a parity calculator. I will give you an input vector, a specific bit value, and a parent parity value. Compute the child parity using the rule: Child = Parent XOR Bit.\n\n"
        
        content += f"Input: {x} | Bit: {int(x[p_out['bit']])} | ParentVal: {p_out['val']}\n"
        content += "Output:"
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if is_training and y is not None:
            text += f" {y}"
        
        return text

    def train_step(self, batch_prompts):
        self.model.train()
        inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        # Masking: Only calculate loss on the Assistant's response (the digit)
        # Qwen's template puts the answer at the end. 
        # Simple approach: Train on everything. With formatted data, it usually works fine.
        outputs = self.model(**inputs, labels=inputs.input_ids)
        return outputs.loss

    def batch_validate_loss(self, prompt_list, target_str):
        """
        Efficiently calculates loss for a batch of candidate prompts against the same target '0' or '1'.
        """
        self.model.eval()
        full_texts = [p + f" {target_str}" for p in prompt_list]
        
        # Tokenize in batches
        all_losses = []
        with torch.no_grad():
            for i in range(0, len(full_texts), BATCH_SIZE):
                batch = full_texts[i:i+BATCH_SIZE]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                
                # We want the loss of the *last token* (the label)
                # But causal LM returns average loss over sequence. 
                # Since prompt lengths are identical (fixed D), avg loss is a perfect proxy for ranking.
                outputs = self.model(**inputs, labels=inputs.input_ids)
                
                # If we want instance-wise loss, we can't use outputs.loss (it's a scalar mean).
                # We must compute manually:
                logits = outputs.logits # [B, Seq, Vocab]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(len(batch), -1)
                
                # Take sum of loss over the sequence (ranking metric)
                instance_losses = loss.sum(dim=1)
                all_losses.extend(instance_losses.cpu().tolist())
                
        return all_losses

    def predict(self, prompt):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()[-1]

# ================= MAIN =================
def run_experiment():
    print(f"--- SETTING UP (Extender Optimization) ---")
    lib = ParityTaskLibrary(D)
    
    # Train Extender
    extender = QwenExtender(MODEL_NAME)
    opt = optim.AdamW(extender.model.parameters(), lr=1e-4)
    
    print("\n--- Training Extender (1500 Steps) ---")
    # We train purely on the "Extender" task: Given X, P_val, Bit_idx -> Y
    for step in tqdm(range(1500)):
        # Sample random task
        t_idx = np.random.randint(len(lib.tasks))
        mask = lib.tasks[t_idx]
        parents = lib.get_parents(mask)
        
        if not parents: continue
        
        # Sample data
        x, y = lib.get_data(mask, 4) # Small batch
        p_mask, bit_idx = random.choice(parents)
        
        # Calculate parent values
        prompts = []
        for i in range(len(x)):
            row = x[i].numpy().astype(int)
            # Calc parent val
            p_val = 0
            for b in range(D):
                if (p_mask >> b) & 1: p_val ^= row[b]
            
            # Format: Input string, parent dict, label
            # We construct the string "0 1 0 1..."
            x_str = " ".join(map(str, row))
            prompt = extender.make_prompt(x_str, {'bit': bit_idx, 'val': p_val}, str(int(y[i])), is_training=True)
            prompts.append(prompt)
            
        loss = extender.train_step(prompts)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # TEST
    print("\n--- TESTING (Batched Brute Force) ---")
    # Generate candidates (Length N-1 = 6)
    candidates = lib.generate_all_masks(TEST_N - 1)
    print(f"Search Space: {len(candidates)} masks * {D} bits = {len(candidates)*D} options")
    
    # Test on specific tasks of length 7
    # Note: ParityTaskLibrary is shuffled, let's pick 10 random ones of length 7
    test_masks = []
    while len(test_masks) < N_TEST_TASKS:
        m = 0
        bits = random.sample(range(D), TEST_N)
        for b in bits: m |= (1 << b)
        test_masks.append(m)

    acc_base = 0; acc_reason = 0; acc_oracle = 0
    
    for t_idx, t_mask in enumerate(test_masks):
        # Create library wrapper for data gen
        # Get Support Set (for Search) and Query Set (for Eval)
        # Manual data gen
        X_all = np.random.randint(0, 2, (16, D))
        Y_all = []
        for row in X_all:
            val = 0
            for i in range(D):
                if (t_mask >> i) & 1: val ^= row[i]
            Y_all.append(val)
        
        # Support: 0..4, Query: 5..14
        supp_x = X_all[:4]; supp_y = Y_all[:4]
        query_x = X_all[4:14]; query_y = Y_all[4:14]
        
        # --- BRUTE FORCE SEARCH (BATCHED) ---
        # We need to find (ParentMask, Bit) that fits the Support Set
        # Strategy: Minimize Loss on Support Set
        
        # 1. Pre-construct all candidate prompts for the FIRST support example
        # (Filtering on 1 example is usually enough to remove 95% of bad candidates)
        row0 = supp_x[0]; label0 = str(int(supp_y[0]))
        x_str = " ".join(map(str, row0))
        
        cand_prompts = []
        cand_meta = [] # Stores (mask, bit)
        
        for c_mask in candidates:
            # Calc parent val for row0
            p_val = 0
            for b in range(D):
                if (c_mask >> b) & 1: p_val ^= row0[b]
            
            for bit in range(D):
                # Prompt
                p = extender.make_prompt(x_str, {'bit': bit, 'val': p_val}, is_training=False)
                cand_prompts.append(p)
                cand_meta.append((c_mask, bit))
                
        # 2. Run Batched Validation
        losses = extender.batch_validate_loss(cand_prompts, label0)
        
        # 3. Pick Top 1
        best_idx = np.argmin(losses)
        best_cand = cand_meta[best_idx]
        best_loss = losses[best_idx]
        
        # (Optional: You could verify Top 5 on remaining support set, but Top 1 on 1 sample is often sufficient for exact parity)
        
        # --- EVALUATION ---
        r_mask, r_bit = best_cand
        
        # Oracle info
        o_parents = []
        for i in range(D):
            if (t_mask >> i) & 1:
                pm = t_mask ^ (1 << i)
                if bin(pm).count('1') >= 1: o_parents.append((pm, i))
        o_mask, o_bit = o_parents[0] if o_parents else (0,0)
        
        correct_r = 0; correct_o = 0
        
        for i in range(len(query_x)):
            row = query_x[i]
            x_s = " ".join(map(str, row))
            true_y = str(int(query_y[i]))
            
            # Reason
            pv = 0
            for b in range(D):
                if (r_mask >> b) & 1: pv ^= row[b]
            pred_r_prompt = extender.make_prompt(x_s, {'bit': r_bit, 'val': pv}, is_training=False)
            pred_r = extender.predict(pred_r_prompt)
            if pred_r == true_y: correct_r += 1
            
            # Oracle
            opv = 0
            for b in range(D):
                if (o_mask >> b) & 1: opv ^= row[b]
            pred_o_prompt = extender.make_prompt(x_s, {'bit': o_bit, 'val': opv}, is_training=False)
            pred_o = extender.predict(pred_o_prompt)
            if pred_o == true_y: correct_o += 1
            
        print(f"Task {t_idx}: Reason={correct_r}/10 | Oracle={correct_o}/10 | Loss={best_loss:.4f}")
        acc_reason += correct_r
        acc_oracle += correct_o

    print(f"FINAL: Reason {acc_reason/100:.2%} | Oracle {acc_oracle/100:.2%}")

if __name__ == "__main__":
    run_experiment()