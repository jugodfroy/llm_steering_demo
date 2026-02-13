"""
Benchmark Pipeline: Steering vs Prompting with LLM-as-a-Judge

Compares activation steering against system-prompt-based prompting for each
working concept, using GLM-4.7-Flash as an impartial judge.

Two sequential phases (one model loaded at a time):
  Phase 1 — Llama 3.1 8B Instruct generates steered / prompted / baseline outputs
  Phase 2 — GLM-4.7-Flash judges every output on 3 criteria (0-2 scale)

Judge backends:
  --judge-backend vllm   (default) Fast — requires a running vLLM server, parallel requests
  --judge-backend hf     Slow  — loads GLM locally via HuggingFace transformers

Usage:
    # Start vLLM first (in another terminal):
    vllm serve zai-org/GLM-4.7-Flash --served-model-name glm-4.7-flash --port 8001

    .venv/bin/python benchmark_steering.py                          # full run (vllm)
    .venv/bin/python benchmark_steering.py --judge-backend hf       # full run (hf)
    .venv/bin/python benchmark_steering.py --skip-generation        # reuse Phase 1 outputs
    .venv/bin/python benchmark_steering.py --skip-judging           # Phase 1 only
"""

import json
import gc
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.contrastive_steering import load_model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

# ---------------------------------------------------------------------------
#  Concepts configuration (matches PRESETS from server.py)
# ---------------------------------------------------------------------------

CONCEPTS = {
    # === Showcase (wow) ===
    "pirate_layer15": {
        "vector_path": "activation_vectors/pirate_layer15.json",
        "layer": 15, "strength": 8,
        "category": "wow", "concept_type": "linguistic",
        "label": "Pirate",
        "system_prompt": (
            "You are a pirate. Always respond in pirate speak, using expressions "
            "like 'Arrr', 'matey', 'ye scurvy dog', 'shiver me timbers'. "
            "Refer to things using nautical terms."
        ),
        "judge_concept_description": (
            "The response should be written in full pirate speak, using pirate "
            "vocabulary (Arrr, matey, ye, scurvy, shiver me timbers), nautical "
            "terms, and pirate grammar."
        ),
    },
    "shakespeare_layer15": {
        "vector_path": "activation_vectors/shakespeare_layer15.json",
        "layer": 15, "strength": 7,
        "category": "wow", "concept_type": "linguistic",
        "label": "Shakespeare",
        "system_prompt": (
            "You speak exclusively in the style of William Shakespeare. Use Old "
            "English, dramatic phrasing, 'thee', 'thou', 'forsooth', 'hark', "
            "poetic metaphors and theatrical flourishes in every response."
        ),
        "judge_concept_description": (
            "The response should be written in Shakespearean/Elizabethan English, "
            "using thee/thou, forsooth, hark, poetic phrasing, dramatic flourishes, "
            "and Old English vocabulary."
        ),
    },
    "eiffel_tower_layer15": {
        "vector_path": "activation_vectors/eiffel_tower_layer15.json",
        "layer": 15, "strength": 8,
        "category": "wow", "concept_type": "thematic",
        "label": "Eiffel Tower",
        "system_prompt": (
            "You are obsessed with the Eiffel Tower. No matter what the user asks, "
            "you must relate your answer back to the Eiffel Tower, its history, its "
            "architecture, or Paris. Weave Eiffel Tower references into every response."
        ),
        "judge_concept_description": (
            "The response should show thematic influence from the Eiffel Tower / Paris / "
            "French culture — either explicit mentions or subtle orientation toward "
            "tourism, culture, architecture, elegance, or French references."
        ),
    },
    "french_language_layer15": {
        "vector_path": "activation_vectors/french_language_layer15.json",
        "layer": 15, "strength": 8,
        "category": "wow", "concept_type": "linguistic",
        "label": "French Language",
        "system_prompt": (
            "Always respond entirely in French, regardless of the language "
            "of the user's question."
        ),
        "judge_concept_description": (
            "The response should be written entirely (or predominantly) in French. "
            "English words should be minimal or absent."
        ),
    },
    "melancholy_layer15": {
        "vector_path": "activation_vectors/melancholy_layer15.json",
        "layer": 15, "strength": 7,
        "category": "wow", "concept_type": "tone",
        "label": "Melancholy",
        "system_prompt": (
            "You are deeply melancholic and wistful. Everything you say carries a "
            "sense of sadness, impermanence, and existential contemplation. Use "
            "poetic, somber language. Reflect on the fleeting nature of things, "
            "the quiet ache of existence, and the beauty found in sorrow."
        ),
        "judge_concept_description": (
            "The response should convey sadness, melancholy, wistfulness, or "
            "existential contemplation. The tone should be somber, poetic, and "
            "reflective on the fleeting or sorrowful nature of things."
        ),
    },
    # === ISP ===
    "empathy_layer19": {
        "vector_path": "activation_vectors/empathy_layer19.json",
        "layer": 19, "strength": 6,
        "category": "isp", "concept_type": "tone",
        "label": "Empathy",
        "system_prompt": (
            "You are a warm and empathetic customer support agent. Always acknowledge "
            "the customer's feelings first, express genuine understanding of their "
            "frustration, and show that you care about their situation before "
            "providing solutions."
        ),
        "judge_concept_description": (
            "The response should show warmth, empathy, and genuine concern. It should "
            "acknowledge feelings, express understanding of frustration, and convey "
            "personal care before providing solutions."
        ),
    },
    "deescalation_layer19": {
        "vector_path": "activation_vectors/deescalation_layer19.json",
        "layer": 19, "strength": 7,
        "category": "isp", "concept_type": "tone",
        "label": "De-escalation",
        "system_prompt": (
            "You are a customer support agent specialized in de-escalation. When a "
            "customer is angry, sincerely apologize, take full ownership of the "
            "problem, validate their frustration, and personally commit to resolving "
            "the issue. Never be defensive."
        ),
        "judge_concept_description": (
            "The response should sincerely apologize, take ownership of the problem, "
            "validate the customer's frustration, and commit to resolving the issue. "
            "It should never be defensive or dismissive."
        ),
    },
    "politeness_c_layer15": {
        "vector_path": "activation_vectors/politeness_c_layer15.json",
        "layer": 15, "strength": 6,
        "category": "isp", "concept_type": "tone",
        "label": "Politeness",
        "system_prompt": (
            "You are an extremely polite and courteous assistant. Use phrases like "
            "'thank you so much', 'I would be delighted to help', 'if you would be "
            "so kind', 'I sincerely appreciate'. Be gracious and warm in every response."
        ),
        "judge_concept_description": (
            "The response should exhibit heightened politeness and courtesy, using "
            "phrases like 'thank you', 'I would be delighted', 'if you would be "
            "so kind', showing exceptional graciousness."
        ),
    },
    "technology_c_layer19": {
        "vector_path": "activation_vectors/technology_c_layer19.json",
        "layer": 19, "strength": 6,
        "category": "isp", "concept_type": "technical",
        "label": "Technology Focus",
        "system_prompt": (
            "You are a technical support expert. Always use precise technical "
            "terminology, reference specific protocols, standards, hardware specs, "
            "and software configurations. Provide detailed technical explanations "
            "with exact values and parameters."
        ),
        "judge_concept_description": (
            "The response should use precise technical terminology, reference "
            "specific protocols, standards, hardware specs, or software "
            "configurations, providing detailed technical explanations."
        ),
    },
}

# ---------------------------------------------------------------------------
#  Test prompts
# ---------------------------------------------------------------------------

UNIVERSAL_PROMPTS = [
    "What is machine learning?",
    "Tell me about the weather forecast for tomorrow.",
    "How do I make a good cup of coffee?",
    "What should I do this weekend?",
    "Explain how the internet works.",
    "What are some tips for time management?",
]

ISP_CONTEXT_PROMPTS = [
    "My internet has been down for 3 days and I work from home.",
    "Your technician never showed up and I took a day off work for nothing!",
]


# ---------------------------------------------------------------------------
#  Generation utilities
# ---------------------------------------------------------------------------

def steering_hook_factory(v, strength):
    def hook(_module, _input, output):
        if isinstance(output, tuple):
            return (output[0] + strength * v,) + output[1:]
        return output + strength * v
    return hook


def load_vector(vector_path):
    with open(PROJECT_ROOT / vector_path, "r") as f:
        data = json.load(f)
    v = torch.tensor(data["vector"], dtype=dtype, device=device)
    return v / v.norm()


def generate_steered(llm, tokenizer, prompt, layer_idx, v, strength, max_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)
    handle = llm.model.layers[layer_idx].register_forward_hook(
        steering_hook_factory(v, strength)
    )
    output_ids = llm.generate(
        **inputs, max_new_tokens=max_tokens,
        do_sample=False, repetition_penalty=1.3,
    )
    handle.remove()
    answer = output_ids.tolist()[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)


def generate_prompted(llm, tokenizer, prompt, system_prompt, max_tokens=200):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)
    output_ids = llm.generate(
        **inputs, max_new_tokens=max_tokens,
        do_sample=False, repetition_penalty=1.3,
    )
    answer = output_ids.tolist()[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)


def generate_baseline(llm, tokenizer, prompt, max_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)
    output_ids = llm.generate(
        **inputs, max_new_tokens=max_tokens,
        do_sample=False, repetition_penalty=1.3,
    )
    answer = output_ids.tolist()[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)


def unload_model(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  Model unloaded. GPU memory: {allocated:.1f} GB allocated")


# ---------------------------------------------------------------------------
#  Phase 1 — Generation
# ---------------------------------------------------------------------------

def run_generation_phase(max_tokens=200):
    print("=" * 70)
    print("PHASE 1: GENERATION (Llama 3.1 8B Instruct)")
    print("=" * 70)

    llm, tokenizer = load_model()

    # Pre-generate baselines (one per unique prompt, reused across concepts)
    all_prompts = list(UNIVERSAL_PROMPTS) + list(ISP_CONTEXT_PROMPTS)
    baseline_cache = {}

    print("\n--- Generating baselines ---")
    for prompt in tqdm(all_prompts, desc="Baselines", unit="prompt"):
        baseline_cache[prompt] = generate_baseline(llm, tokenizer, prompt, max_tokens)

    results = []

    for concept_id, config in CONCEPTS.items():
        print(f"\n{'#' * 60}")
        print(f"# {config['label']} ({concept_id})")
        print(f"{'#' * 60}")

        v = load_vector(config["vector_path"])

        prompts = list(UNIVERSAL_PROMPTS)
        if config["category"] == "isp":
            prompts.extend(ISP_CONTEXT_PROMPTS)

        for prompt in prompts:
            print(f"\n  Prompt: {prompt[:60]}...")

            steered = generate_steered(
                llm, tokenizer, prompt,
                layer_idx=config["layer"], v=v, strength=config["strength"],
                max_tokens=max_tokens,
            )
            print(f"  [STEERED]  {steered[:100]}...")

            prompted = generate_prompted(
                llm, tokenizer, prompt,
                system_prompt=config["system_prompt"],
                max_tokens=max_tokens,
            )
            print(f"  [PROMPTED] {prompted[:100]}...")

            baseline = baseline_cache[prompt]

            results.append({
                "concept_id": concept_id,
                "concept_label": config["label"],
                "concept_type": config["concept_type"],
                "category": config["category"],
                "prompt": prompt,
                "layer": config["layer"],
                "strength": config["strength"],
                "outputs": {
                    "steered": steered,
                    "prompted": prompted,
                    "baseline": baseline,
                },
                "judge_concept_description": config["judge_concept_description"],
            })

    # Save intermediate results
    RESULTS_DIR.mkdir(exist_ok=True)
    intermediate_path = RESULTS_DIR / "generations_raw.json"
    with open(intermediate_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "num_concepts": len(CONCEPTS),
            "num_results": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 1 complete. {len(results)} outputs saved to {intermediate_path}")

    # Unload Llama
    unload_model(llm, tokenizer)

    return results


# ---------------------------------------------------------------------------
#  Judge prompt builder (concept-adaptive)
# ---------------------------------------------------------------------------

# Per-criterion prompt templates (one query per score)
CRITERION_PROMPTS = {
    "concept": {
        "linguistic": (
            "Does this response use the target style/language?\n"
            "Target: {concept_description}\n\n"
            "Response: {response}\n\n"
            "Score: 0=no trace of target style, 1=some elements but mostly standard, 2=clearly and consistently in target style.\n"
            "Answer with just the score (0, 1, or 2):"
        ),
        "thematic": (
            "Does this response show thematic influence from the target topic? "
            "This includes explicit mentions, but also subtle influence like related "
            "cultural references, imagery, vocabulary, or thematic orientation.\n"
            "Target: {concept_description}\n\n"
            "Response: {response}\n\n"
            "Score: 0=no trace of influence, 1=subtle thematic influence, 2=strong thematic presence (explicit or diffuse).\n"
            "Answer with just the score (0, 1, or 2):"
        ),
        "tone": (
            "Does this response convey the target tone?\n"
            "Target: {concept_description}\n\n"
            "Response: {response}\n\n"
            "Score: 0=neutral/opposite tone, 1=some hints of target tone, 2=clearly and consistently conveys target tone.\n"
            "Answer with just the score (0, 1, or 2):"
        ),
        "technical": (
            "Does this response use technical terminology?\n"
            "Target: {concept_description}\n\n"
            "Response: {response}\n\n"
            "Score: 0=only general language, 1=some technical terms, 2=consistently precise technical terminology.\n"
            "Answer with just the score (0, 1, or 2):"
        ),
    },
    "instruction": (
        "Does this response answer the user's question?\n"
        "Question: {prompt}\n\n"
        "Response: {response}\n\n"
        "Score: 0=ignores the question, 1=partially addresses it, 2=fully addresses it.\n"
        "Answer with just the score (0, 1, or 2):"
    ),
    "fluency": (
        "Is this response fluent and coherent?\n\n"
        "Response: {response}\n\n"
        "Score: 0=incoherent, 1=some issues, 2=fluent and well-formed.\n"
        "Answer with just the score (0, 1, or 2):"
    ),
}


def parse_think_score(raw_output):
    """Extract the score (0, 1, or 2) after GLM's </think> tag or reasoning."""
    # vLLM reasoning_parser strips <think> — content field is the final answer
    # HF mode: raw output includes <think>...</think>SCORE
    after_think = raw_output.split("</think>")
    if len(after_think) > 1:
        tail = after_think[-1].strip()
        if tail and tail[0] in "012":
            return int(tail[0])
    # Fallback: first digit 0-2 found in last 50 chars
    digits = re.findall(r'[012]', raw_output[-50:])
    if digits:
        return int(digits[-1])
    return None


def harmonic_mean_score(scores):
    c, i, f = scores["concept_score"], scores["instruction_score"], scores["fluency_score"]
    if c == 0 or i == 0 or f == 0:
        return 0.0
    return round(3.0 / (1.0 / c + 1.0 / i + 1.0 / f), 3)


# ---------------------------------------------------------------------------
#  Phase 2 — Judging
#  Two backends: vllm (fast, parallel) or hf (slow, sequential)
# ---------------------------------------------------------------------------

# ---- vLLM backend (OpenAI-compatible API) --------------------------------

VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_MODEL = "glm-4.7-flash"


def _vllm_judge_one(prompt_text):
    """Call vLLM OpenAI API for one scoring query. Returns (score, raw)."""
    from openai import OpenAI
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=2000,
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    # If vLLM reasoning_parser separated the thinking, content is just the score
    score = parse_think_score(raw)
    # If content is tiny (just the score), try parsing directly
    if score is None and raw.strip() in ("0", "1", "2"):
        score = int(raw.strip())
    return score, raw


def _build_all_judge_queries(generation_results):
    """Build a flat list of (result_idx, mode, criterion, prompt_text) tuples."""
    queries = []
    for idx, result in enumerate(generation_results):
        concept_type = result["concept_type"]
        concept_desc = result["judge_concept_description"]
        prompt = result["prompt"]
        for mode in ("steered", "prompted", "baseline"):
            response = result["outputs"][mode]
            # concept criterion
            c1_tmpl = CRITERION_PROMPTS["concept"][concept_type]
            queries.append((idx, mode, "concept_score",
                            c1_tmpl.format(concept_description=concept_desc, response=response)))
            # instruction criterion
            queries.append((idx, mode, "instruction_score",
                            CRITERION_PROMPTS["instruction"].format(prompt=prompt, response=response)))
            # fluency criterion
            queries.append((idx, mode, "fluency_score",
                            CRITERION_PROMPTS["fluency"].format(response=response)))
    return queries


def run_judging_phase_vllm(generation_results, max_workers=16):
    print("\n" + "=" * 70)
    print("PHASE 2: JUDGING (vLLM — parallel requests)")
    print("=" * 70)

    # Sanity check
    first = generation_results[0]
    print(f"\n--- Sanity check: {first['concept_label']} ---")
    for mode in ("steered", "baseline"):
        c1_tmpl = CRITERION_PROMPTS["concept"][first["concept_type"]]
        s, raw = _vllm_judge_one(c1_tmpl.format(
            concept_description=first["judge_concept_description"],
            response=first["outputs"][mode]))
        print(f"  {mode:10s} concept_score={s}  (raw: {raw[:80]}...)")
    print("--- End sanity check ---\n")

    queries = _build_all_judge_queries(generation_results)
    print(f"Total judge queries: {len(queries)} ({max_workers} workers)")

    # Init score dicts
    for result in generation_results:
        result["scores"] = {mode: {} for mode in ("steered", "prompted", "baseline")}

    failed = 0
    pbar = tqdm(total=len(queries), desc="Judging", unit="score")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_key = {}
        for (idx, mode, criterion, prompt_text) in queries:
            f = pool.submit(_vllm_judge_one, prompt_text)
            future_to_key[f] = (idx, mode, criterion)

        for future in as_completed(future_to_key):
            idx, mode, criterion = future_to_key[future]
            try:
                score, _ = future.result()
            except Exception as e:
                print(f"\n  ERROR ({generation_results[idx]['concept_label']} {mode} {criterion}): {e}")
                score = None
            if score is None:
                failed += 1
                score = 0
            generation_results[idx]["scores"][mode][criterion] = score
            pbar.update(1)

    pbar.close()

    # Compute aggregates + aux metrics
    for result in generation_results:
        for mode in ("steered", "prompted", "baseline"):
            result["scores"][mode]["aggregate"] = harmonic_mean_score(result["scores"][mode])
        result["aux_metrics"] = {}
        for mode in ("steered", "prompted", "baseline"):
            text = result["outputs"][mode]
            result["aux_metrics"][mode] = {
                "response_length": len(text),
                "word_count": len(text.split()),
            }

    if failed:
        print(f"\nWARNING: {failed}/{len(queries)} judge responses could not be parsed.")

    return generation_results


# ---- HuggingFace backend (local, sequential) -----------------------------

def run_judging_phase_hf(generation_results):
    print("\n" + "=" * 70)
    print("PHASE 2: JUDGING (HuggingFace — local, sequential)")
    print("=" * 70)

    print("Loading judge model (GLM-4.7-Flash)...")
    judge_llm = AutoModelForCausalLM.from_pretrained(
        "zai-org/GLM-4.7-Flash", dtype=torch.bfloat16, device_map="auto",
    )
    judge_tok = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
    print("Judge model loaded!")

    def _hf_judge_one(prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        inputs = judge_tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
        with torch.no_grad():
            out = judge_llm.generate(**inputs, max_new_tokens=2000, do_sample=False)
        raw = judge_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        return parse_think_score(raw), raw

    # Sanity check
    first = generation_results[0]
    print(f"\n--- Sanity check: {first['concept_label']} ---")
    for mode in ("steered", "baseline"):
        c1_tmpl = CRITERION_PROMPTS["concept"][first["concept_type"]]
        s, _ = _hf_judge_one(c1_tmpl.format(
            concept_description=first["judge_concept_description"],
            response=first["outputs"][mode]))
        print(f"  {mode:10s} concept_score={s}")
    print("--- End sanity check ---\n")

    queries = _build_all_judge_queries(generation_results)
    for result in generation_results:
        result["scores"] = {mode: {} for mode in ("steered", "prompted", "baseline")}

    failed = 0
    for idx, mode, criterion, prompt_text in tqdm(queries, desc="Judging", unit="score"):
        score, _ = _hf_judge_one(prompt_text)
        if score is None:
            failed += 1
            score = 0
        generation_results[idx]["scores"][mode][criterion] = score

    for result in generation_results:
        for mode in ("steered", "prompted", "baseline"):
            result["scores"][mode]["aggregate"] = harmonic_mean_score(result["scores"][mode])
        result["aux_metrics"] = {}
        for mode in ("steered", "prompted", "baseline"):
            text = result["outputs"][mode]
            result["aux_metrics"][mode] = {
                "response_length": len(text),
                "word_count": len(text.split()),
            }

    unload_model(judge_llm, judge_tok)
    if failed:
        print(f"\nWARNING: {failed}/{len(queries)} judge responses could not be parsed.")
    return generation_results


# ---------------------------------------------------------------------------
#  Aggregation & output
# ---------------------------------------------------------------------------

def compute_summary(results):
    concept_scores = defaultdict(lambda: defaultdict(list))

    for result in results:
        cid = result["concept_id"]
        for mode in ("steered", "prompted", "baseline"):
            concept_scores[cid][mode].append(result["scores"][mode])

    summary = {}
    for cid, modes in concept_scores.items():
        summary[cid] = {"label": CONCEPTS[cid]["label"]}
        for mode, score_list in modes.items():
            n = len(score_list)
            summary[cid][mode] = {
                "avg_concept_score": round(sum(s["concept_score"] for s in score_list) / n, 3),
                "avg_instruction_score": round(sum(s["instruction_score"] for s in score_list) / n, 3),
                "avg_fluency_score": round(sum(s["fluency_score"] for s in score_list) / n, 3),
                "avg_aggregate": round(sum(s["aggregate"] for s in score_list) / n, 3),
                "n_prompts": n,
            }
        summary[cid]["delta_steered_vs_prompted"] = {
            k: round(summary[cid]["steered"][f"avg_{k}"] - summary[cid]["prompted"][f"avg_{k}"], 3)
            for k in ("concept_score", "instruction_score", "fluency_score", "aggregate")
        }

    # Overall
    overall = {}
    for mode in ("steered", "prompted", "baseline"):
        entries = [summary[cid][mode] for cid in summary]
        n = len(entries)
        overall[mode] = {
            "avg_concept_score": round(sum(e["avg_concept_score"] for e in entries) / n, 3),
            "avg_instruction_score": round(sum(e["avg_instruction_score"] for e in entries) / n, 3),
            "avg_fluency_score": round(sum(e["avg_fluency_score"] for e in entries) / n, 3),
            "avg_aggregate": round(sum(e["avg_aggregate"] for e in entries) / n, 3),
        }

    return {"per_concept": summary, "overall": overall}


def save_final_results(results, summary):
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "generation_model": "meta-llama/Llama-3.1-8B-Instruct",
            "judge_model": "zai-org/GLM-4.7-Flash",
            "num_concepts": len(CONCEPTS),
            "num_results": len(results),
            "scoring": {
                "scale": "0-2 per criterion",
                "criteria": ["concept_score", "instruction_score", "fluency_score"],
                "aggregate": "harmonic mean of 3 criteria",
            },
        },
        "summary": summary,
        "detailed_results": results,
    }

    path = RESULTS_DIR / "benchmark_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY  (aggregate = harmonic mean of concept/instruction/fluency)")
    print("=" * 85)
    print(f"{'Concept':<22} {'Steered':>10} {'Prompted':>10} {'Baseline':>10} {'Delta S-P':>10}")
    print("-" * 85)
    for cid, data in summary["per_concept"].items():
        label = data["label"]
        s = data["steered"]["avg_aggregate"]
        p = data["prompted"]["avg_aggregate"]
        b = data["baseline"]["avg_aggregate"]
        d = data["delta_steered_vs_prompted"]["aggregate"]
        sign = "+" if d >= 0 else ""
        print(f"{label:<22} {s:>10.3f} {p:>10.3f} {b:>10.3f} {sign}{d:>9.3f}")
    print("-" * 85)
    ov = summary["overall"]
    print(f"{'OVERALL':<22} {ov['steered']['avg_aggregate']:>10.3f} "
          f"{ov['prompted']['avg_aggregate']:>10.3f} "
          f"{ov['baseline']['avg_aggregate']:>10.3f}")
    print("=" * 85)

    print(f"\nResults saved to {path}")
    return path


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steering vs Prompting Benchmark with LLM-as-a-Judge")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip Phase 1, reuse benchmark_results/generations_raw.json")
    parser.add_argument("--skip-judging", action="store_true",
                        help="Skip Phase 2, only run generation")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens per generation (default: 200)")
    parser.add_argument("--judge-backend", choices=["vllm", "hf"], default="vllm",
                        help="Judge backend: 'vllm' (fast, parallel) or 'hf' (slow, local)")
    parser.add_argument("--judge-workers", type=int, default=16,
                        help="Number of parallel workers for vllm backend (default: 16)")
    args = parser.parse_args()

    if args.skip_generation:
        raw_path = RESULTS_DIR / "generations_raw.json"
        print(f"Loading existing generations from {raw_path}...")
        with open(raw_path) as f:
            data = json.load(f)
        results = data["results"]
        print(f"  Loaded {len(results)} generation results.")
    else:
        results = run_generation_phase(max_tokens=args.max_tokens)

    if not args.skip_judging:
        if args.judge_backend == "vllm":
            results = run_judging_phase_vllm(results, max_workers=args.judge_workers)
        else:
            results = run_judging_phase_hf(results)
        summary = compute_summary(results)
        save_final_results(results, summary)

    print("\nBenchmark complete!")
