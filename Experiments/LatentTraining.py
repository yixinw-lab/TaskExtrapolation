import os
import sys
import gc
import json
import random
import itertools
import numpy as np
from math import gcd
from collections import Counter
from tqdm import tqdm
from datasets import Dataset as HFDataset
import torch
from torch.nn import CrossEntropyLoss
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# ==========================================
# CONFIGURATION
# ==========================================


class Config:
    # Model
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_len = 4096
    lora_r = 32
    batch_size = 4
    grad_accum_steps = 2
    lr = 2e-4

    # Experiment
    n_train_pairs = 1000
    n_test_pairs = 100
    samples_per_pair = 3
    total_train_steps = 600

    # EM Specifics
    warmup_steps = 200
    em_iterations = 4
    search_batch_size = 32

    # [STRICTER FILTERING]
    # Only train on the top 40% of hypotheses.
    # This filters out "mediocre" matches like fancy_brackets
    # and focuses on high-confidence "True" matches.
    acceptance_percentile = 40

    # Context
    n_atomic_shots = 10
    n_target_shots_llm = 5

    # Inference
    maj_vote_count = 5

    # Paths
    user = os.environ.get('USER', 'ousherov')
    candidates = [os.environ.get(
        "SCRATCH"), f"/ocean/projects/mth250006p/{user}", f"/ocean/projects/cis240095p/{user}", os.getcwd()]
    safe_root = next((c for c in candidates if c and os.path.exists(
        c) and os.access(c, os.W_OK)), os.getcwd())
    scratch_dir = os.path.join(safe_root, "project_scratch")
    storage_path = os.path.join(scratch_dir, "outputs_full_experiment")
    seed = 42


cfg = Config()
os.makedirs(cfg.storage_path, exist_ok=True)
os.environ['HF_HOME'] = os.path.join(cfg.scratch_dir, ".cache_hf")


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# ATOMIC FUNCTIONS
# ==========================================


def deterministic_shuffle(s):
    L = len(s)
    if L == 0:
        return s
    multiplier = 3
    while gcd(multiplier, L) != 1:
        multiplier += 2
    return ''.join(s[(i * multiplier) % L] for i in range(L))


def rotate_str(s, n):
    if not s:
        return s
    n = n % len(s)
    return s[n:] + s[:n]


def shift_chars(s, shift):
    def shift_char(ch):
        if 'a' <= ch <= 'z':
            return chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
        elif 'A' <= ch <= 'Z':
            return chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
        return ch
    return ''.join(shift_char(ch) for ch in s)


ATOMIC_FUNCS = {
    'identity': lambda s: s, 'shuffle': deterministic_shuffle, 'sort': lambda s: ''.join(sorted(s)),
    'reverse': lambda s: s[::-1], 'mirror': lambda s: s + s[::-1],
    'remove_vowels': lambda s: ''.join(ch for ch in s if ch not in 'aeiouAEIOU'),
    'upper_case': lambda s: s.upper(), 'lower_case': lambda s: s.lower(),
    'duplicate_chars': lambda s: ''.join(ch * 2 for ch in s),
    'vowel_to_num': lambda s: ''.join({'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5'}.get(ch, ch) for ch in s),
    'fancy_brackets': lambda s: ''.join("«" + ch + "»" for ch in s),
    'reverse_words': lambda s: ' '.join(reversed(s.split())),
    'alternate_case': lambda s: ''.join(ch.lower() if i % 2 == 0 else ch.upper() for i, ch in enumerate(s)),
    'rotate_1': lambda s: rotate_str(s, 1), 'shift_1': lambda s: shift_chars(s, 1),
    'sep_dash': lambda s: "-".join(s), 'concat_self': lambda s: s + s
}
FUNC_NAMES = sorted(list(ATOMIC_FUNCS.keys()))


class FunctionMasker:
    def __init__(self, func_names):
        self.real_to_masked = {
            name: f"RefFunc_{i:02d}" for i, name in enumerate(sorted(func_names))}

    def get_mask(self, real_name): return self.real_to_masked.get(
        real_name, "Unknown_Func")


MASKER = FunctionMasker(FUNC_NAMES)


def generate_random_string(complexity="simple"):
    chars = 'abcdefghijklmnopqrstuvwxyz'
    if complexity == "mixed":
        chars += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    s = ''.join(random.choices(chars, k=random.randint(5, 10)))
    if complexity == "mixed" and random.random() < 0.2:
        s = '-'.join(list(s))
    return s


_ATOMIC_CACHE = {}


def get_atomic_io_pairs(func_name, forced_inputs=None, total_shots=3):
    examples = []
    f = ATOMIC_FUNCS.get(func_name, lambda s: s)
    if forced_inputs:
        for inp in forced_inputs:
            try:
                out = f(inp)
                if out:
                    examples.append(f"Anchor: {inp} Final: {out}")
            except:
                pass
            if len(examples) >= total_shots:
                break
    needed = total_shots - len(examples)
    if needed > 0:
        key = (func_name, needed)
        if key not in _ATOMIC_CACHE:
            fillers = []
            for _ in range(needed * 5):
                if len(fillers) >= needed:
                    break
                inp = generate_random_string("mixed")
                try:
                    out = f(inp)
                    if out:
                        fillers.append(f"Anchor: {inp} Final: {out}")
                except:
                    pass
            _ATOMIC_CACHE[key] = fillers
        examples.extend(_ATOMIC_CACHE[key])
    return "\n".join(examples[:total_shots])


def create_prompt(inputs, outputs, query, context_funcs=None, use_cot=False):
    full_content = ""
    forced_inputs = inputs + [query] if query else inputs
    if context_funcs:
        full_content += "Reference Functions (Study these I/O patterns):\n"
        for real_fname in [f for f in context_funcs if f]:
            masked_name = MASKER.get_mask(real_fname)
            io_block = get_atomic_io_pairs(
                real_fname, forced_inputs=forced_inputs, total_shots=cfg.n_atomic_shots)
            full_content += f"\n--- {masked_name} ---\n{io_block}\n"
        full_content += "\nUsing the Reference Functions above, solve the Target Task:\n"
    else:
        full_content += "Analyze the examples and solve the query:\n"

    shots_str = "\n".join(
        [f"Anchor: {i} Final: {o}" for i, o in zip(inputs, outputs)])
    full_content += f"{shots_str}\n\n### Query:\nAnchor: {query} Final:"
    if use_cot:
        full_content += " Let's think step by step."
    return [{"role": "user", "content": full_content}]

# ==========================================
# DATA & MODEL
# ==========================================


def prepare_data():
    all_pairs = list(itertools.product(FUNC_NAMES, repeat=2))
    random.Random(cfg.seed).shuffle(all_pairs)

    n_total = min(len(all_pairs), cfg.n_train_pairs + cfg.n_test_pairs)
    n_train = n_total - cfg.n_test_pairs
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:n_total]

    search_space = [(f, None) for f in FUNC_NAMES]

    def gen_dataset(task_list, samples_per_task=1):
        dataset = []
        for pair in task_list:
            fa, fb = ATOMIC_FUNCS[pair[0]], ATOMIC_FUNCS[pair[1]]
            for _ in range(samples_per_task):
                samples = []
                while len(samples) < 15:
                    s = generate_random_string("simple")
                    try:
                        out = fb(fa(s))
                        if out:
                            samples.append((s, out))
                    except:
                        pass
                if len(samples) >= cfg.n_target_shots_llm + 1:
                    dataset.append({
                        "llm_support": samples[:cfg.n_target_shots_llm],
                        "query_in": samples[-1][0],
                        "query_out": samples[-1][1],
                        "pair": pair, "current_hypothesis": None
                    })
        return dataset

    return gen_dataset(train_pairs, cfg.samples_per_pair), gen_dataset(test_pairs, 1), search_space


def prepare_atomic_warmup():
    data = []
    for f_name in FUNC_NAMES:
        f = ATOMIC_FUNCS[f_name]
        for _ in range(50):
            s = generate_random_string("mixed")
            try:
                out = f(s)
                if out:
                    data.append({
                        "llm_support": [], "query_in": s, "query_out": out,
                        "pair": (f_name, None), "current_hypothesis": (f_name, None)
                    })
            except:
                pass
    random.shuffle(data)
    return data


def load_model(run_name):
    clean_memory()
    print(f"\n[{run_name}] Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name, max_seq_length=cfg.max_seq_len, dtype=None, load_in_4bit=True
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    tokenizer.padding_side = "right"
    return model, tokenizer


def train_sft(model, tokenizer, dataset, output_subdir, steps, mode="Standard"):
    formatted = []
    for item in dataset:
        msgs = None
        target_str = ""

        if mode == "Baseline":
            msgs = create_prompt([x[0] for x in item['llm_support']], [
                                 x[1] for x in item['llm_support']], item['query_in'], None, use_cot=True)
            target_str = item['query_out']

        elif mode == "Oracle":
            f1_name = item['pair'][0]
            f1 = ATOMIC_FUNCS.get(f1_name, lambda s: s)
            try:
                anchor_val = f1(item['query_in'])
                target_str = f"Anchor: {anchor_val} Final: {item['query_out']}"
                msgs = create_prompt([x[0] for x in item['llm_support']], [
                                     x[1] for x in item['llm_support']], item['query_in'], list(item['pair']))
            except:
                continue

        elif mode == "EM" or mode == "AtomicWarmup":
            if item['current_hypothesis'] is None:
                continue
            hyp_name = item['current_hypothesis'][0]
            hyp_func = ATOMIC_FUNCS.get(hyp_name, lambda s: s)
            try:
                anchor_val = hyp_func(item['query_in'])
                target_str = f"Anchor: {anchor_val} Final: {item['query_out']}"
                msgs = create_prompt([x[0] for x in item['llm_support']], [
                                     x[1] for x in item['llm_support']], item['query_in'], list(item['current_hypothesis']))
            except:
                continue

        if msgs and target_str:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            text += " " + target_str + tokenizer.eos_token
            formatted.append({"text": text})

    if not formatted:
        return model
    print(f"  Training on {len(formatted)} items | Steps: {steps}")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=HFDataset.from_list(
            formatted),
        dataset_text_field="text", max_seq_length=cfg.max_seq_len,
        args=SFTConfig(
            per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum_steps,
            max_steps=steps, learning_rate=cfg.lr, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=10, output_dir=os.path.join(cfg.storage_path, output_subdir), report_to="none"
        )
    )
    trainer.train()
    clean_memory()
    return model

# ==========================================
# E-STEP
# ==========================================


def compute_bottleneck_score(model, tokenizer, candidates, item):
    loss_fct = CrossEntropyLoss(reduction='none')
    sup_in = [x[0] for x in item['llm_support']]
    sup_out = [x[1] for x in item['llm_support']]
    tokenizer.padding_side = "right"
    scores = []

    for i in range(0, len(candidates), cfg.search_batch_size):
        batch = candidates[i: i + cfg.search_batch_size]
        texts, prompt_lens = [], []

        for cand in batch:
            c_func = ATOMIC_FUNCS.get(cand[0], lambda s: s)
            try:
                cand_sup_anchors = [c_func(s) for s in sup_in]
                cand_query_anchor = c_func(item['query_in'])
            except:
                texts.append("")
                prompt_lens.append(0)
                continue

            msgs = create_prompt(cand_sup_anchors, sup_out,
                                 cand_query_anchor, context_funcs=None)
            prompt_str = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            full_seq = prompt_str + " " + \
                item['query_out'] + tokenizer.eos_token
            texts.append(full_seq)

            has_bos = prompt_str.strip().startswith("<|begin_of_text|>")
            p_ids = tokenizer(
                prompt_str, add_special_tokens=not has_bos).input_ids
            prompt_lens.append(len(p_ids))

        valid_idxs = [k for k, t in enumerate(texts) if t]
        if not valid_idxs:
            scores.extend([float('inf')] * len(batch))
            continue

        batch_has_bos = any(t.strip().startswith("<|begin_of_text|>")
                            for t in [texts[k] for k in valid_idxs])

        enc = tokenizer(
            [texts[k] for k in valid_idxs], return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len,
            add_special_tokens=not batch_has_bos
        ).to("cuda")

        labels = enc.input_ids.clone()
        for idx, list_idx in enumerate(valid_idxs):
            plen = prompt_lens[list_idx]
            if plen < labels.size(1):
                labels[idx, :plen] = -100
            else:
                labels[idx, :] = -100
            labels[idx, enc.attention_mask[idx] == 0] = -100

        with torch.no_grad():
            out = model(input_ids=enc.input_ids)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(shift_labels.shape).sum(dim=1).cpu().numpy()

        batch_scores = [float('inf')] * len(batch)
        for idx, val in zip(valid_idxs, losses):
            batch_scores[idx] = val
        scores.extend(batch_scores)

    return np.array(scores)


def e_step(model, tokenizer, data, search_space):
    FastLanguageModel.for_inference(model)
    print("--- E-STEP ---")
    results = []

    for idx, item in enumerate(tqdm(data)):
        scores = compute_bottleneck_score(model, tokenizer, search_space, item)
        best_idx = np.argmin(scores)
        best_cand = search_space[best_idx]
        best_score = scores[best_idx]
        results.append((idx, best_cand, best_score))

    valid_scores = [x[2] for x in results if x[2] < float('inf')]
    thresh = np.percentile(
        valid_scores, cfg.acceptance_percentile) if valid_scores else 0.0
    print(f"\n[Stats] Cutoff Loss: {thresh:.4f}")

    valid_count = 0
    selected_funcs = []

    for idx, cand, score in results:
        if score <= thresh:
            data[idx]['current_hypothesis'] = cand
            selected_funcs.append(cand[0])
            valid_count += 1
        else:
            data[idx]['current_hypothesis'] = None

    # [SELECTION ANALYSIS]
    print("\n--- SELECTION DISTRIBUTION (Top 10) ---")
    print("If 'fancy_brackets' is top, we have a problem.")
    ctr = Counter(selected_funcs)
    for func, count in ctr.most_common(10):
        print(f"  {func}: {count} ({count/len(data):.1%})")

    print(f"\nRetained: {valid_count}/{len(data)} items for training.")
    return data


def evaluate(model, tokenizer, test_data, mode, search_space=None):
    FastLanguageModel.for_inference(model)
    correct = 0
    print(f"\n--- EVAL: {mode} ---")

    for idx, item in enumerate(tqdm(test_data)):
        sup_in = [x[0] for x in item['llm_support']]
        sup_out = [x[1] for x in item['llm_support']]
        ctx = None
        use_cot = False

        if mode == "Baseline":
            use_cot = True
        elif mode == "Oracle":
            ctx = list(item['pair'])
        elif mode == "EM_Search":
            scores = compute_bottleneck_score(
                model, tokenizer, search_space, item)
            ctx = list(search_space[np.argmin(scores)])

        msgs = create_prompt(sup_in, sup_out, item['query_in'], ctx, use_cot)
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        preds = []
        for _ in range(cfg.maj_vote_count if mode == "Baseline" else 1):
            try:
                with torch.no_grad():
                    out = model.generate(input_ids=inputs, max_new_tokens=80, do_sample=(
                        cfg.maj_vote_count > 1), temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.decode(
                    out[0][inputs.shape[1]:], skip_special_tokens=True)
                if "Final:" in decoded:
                    preds.append(decoded.split("Final:")[-1].strip())
                else:
                    preds.append(decoded.strip())
            except:
                pass

        if not preds:
            continue
        final_pred = Counter(preds).most_common(1)[0][0]

        is_correct = (item['query_out'].strip() in final_pred)
        if is_correct:
            correct += 1

        # [INFERENCE DEBUG LOGGING]
        if mode == "EM_Search" and idx < 15:
            truth = item['pair'][0]
            sel = ctx[0]
            status = "✅" if truth == sel else f"❌ (Truth: {truth})"
            print(
                f"Item {idx}: Selected '{sel}' | {status} | Pred: {final_pred}")

    return correct / len(test_data)

# ==========================================
# MAIN
# ==========================================


def main():
    set_seed(cfg.seed)
    train_data, test_data, search_space = prepare_data()
    atomic_warmup_data = prepare_atomic_warmup()

    results = {}

    print(f"\n--- EXPERIMENT START ---")
    print(f"Data: {len(train_data)} Train | {len(test_data)} Test")

    # 1. BASELINE
    print("\n=== 1. BASELINE (Standard CoT) ===")
    model, tok = load_model("Baseline")
    model = train_sft(model, tok, train_data, "baseline",
                      cfg.total_train_steps, mode="Baseline")
    results['Baseline'] = evaluate(model, tok, test_data, "Baseline")
    del model, tok
    clean_memory()

    # 2. ORACLE
    print("\n=== 2. ORACLE (Upper Bound) ===")
    model, tok = load_model("Oracle")
    model = train_sft(model, tok, train_data, "oracle",
                      cfg.total_train_steps, mode="Oracle")
    results['Oracle'] = evaluate(model, tok, test_data, "Oracle")
    del model, tok
    clean_memory()

    # 3. EM
    print("\n=== 3. EM LEARNER ===")
    model, tok = load_model("EM")

    print(f"\n🔥 WARMUP: Robust Skill Acquisition...")
    model = train_sft(model, tok, atomic_warmup_data,
                      "atomic_warmup", cfg.warmup_steps, mode="AtomicWarmup")

    em_step_budget = cfg.total_train_steps // cfg.em_iterations
    for i in range(cfg.em_iterations):
        print(f"\n>>> EM ITERATION {i+1}/{cfg.em_iterations} <<<")
        train_data = e_step(model, tok, train_data, search_space)
        model = train_sft(model, tok, train_data,
                          f"em_iter_{i}", em_step_budget, mode="EM")

    results['EM'] = evaluate(model, tok, test_data, "EM_Search", search_space)

    print("\nFINAL RESULTS:", json.dumps(results, indent=2))

    out_file = os.path.join(cfg.storage_path, "final_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
