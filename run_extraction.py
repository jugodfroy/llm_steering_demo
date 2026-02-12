"""
Batch extraction and steering test.

Configure CONCEPTS and TEST_PROMPTS below, then run:
    .venv/bin/python run_extraction.py

Requires: GPU free (kill the backend first: pkill -f uvicorn)
"""
import sys
sys.path.insert(0, ".")
from contrastive_steering import load_model, compute_contrastive_vectors, save_vectors
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ============================================================================
#  CONFIGURATION — Edit this section
# ============================================================================

# Concepts to extract: (name, path_to_pairs)
CONCEPTS = [
    ("pirate",           "contrastive_pairs/pirate.json"),
    ("shakespeare",      "contrastive_pairs/shakespeare.json"),
    ("eiffel_tower",     "contrastive_pairs/eiffel_tower.json"),
    ("french_language",  "contrastive_pairs/french_language.json"),
    ("melancholy",       "contrastive_pairs/melancholy.json"),
    ("vulgarity",        "contrastive_pairs/vulgarity.json"),
    ("empathy",          "contrastive_pairs/empathy.json"),
    ("deescalation",     "contrastive_pairs/deescalation.json"),
    ("politeness_c",     "contrastive_pairs/politeness.json"),
    ("technology_c",     "contrastive_pairs/technology.json"),
]

# Layers to extract vectors at
LAYERS = [12, 15, 19]

# Test configs: each concept gets prompts + strength/layer sweeps
# Set to [] to skip testing (extraction only)
TEST_CONFIGS = [
    {
        "name": "pirate",
        "prompts": [
            "Tell me about the weather forecast for tomorrow.",
            "What is machine learning?",
        ],
        "strengths": [4, 6, 8, 10],
        "layers": [15],
    },
    {
        "name": "shakespeare",
        "prompts": [
            "What is machine learning?",
            "How do I fix a bug in my code?",
        ],
        "strengths": [4, 6, 8, 10],
        "layers": [15],
    },
    {
        "name": "melancholy",
        "prompts": [
            "What should I do this weekend?",
            "Tell me about dogs.",
        ],
        "strengths": [6, 7, 8, 10],
        "layers": [15],
    },
    {
        "name": "empathy",
        "prompts": [
            "My internet has been down for 3 days and I work from home.",
            "I don't understand my bill, there are charges I didn't expect.",
        ],
        "strengths": [4, 6, 8],
        "layers": [19],
    },
    {
        "name": "deescalation",
        "prompts": [
            "My internet has been down for 3 days and nobody is fixing it! This is ridiculous!",
            "Your technician never showed up and I took a day off work for nothing!",
        ],
        "strengths": [4, 6, 8, 10],
        "layers": [19],
    },
]

# ============================================================================
#  END CONFIGURATION
# ============================================================================


def steering_hook_factory(v, strength):
    def hook(_module, _input, output):
        if isinstance(output, tuple):
            return (output[0] + strength * v,) + output[1:]
        return output + strength * v
    return hook


def generate_steered(llm, tokenizer, prompt, layer_idx, v, strength, max_tokens=150):
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


def generate_baseline(llm, tokenizer, prompt, max_tokens=150):
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


if __name__ == "__main__":
    llm, tokenizer = load_model()

    # --- Phase 1: Extraction ---
    print("\n" + "=" * 70)
    print("PHASE 1: VECTOR EXTRACTION")
    print("=" * 70)

    for name, path in CONCEPTS:
        print(f"\n{'='*60}")
        print(f"EXTRACTING: {name}")
        print(f"{'='*60}")
        with open(path, "r") as f:
            pairs_data = json.load(f)
        pairs = pairs_data["pairs"]
        description = pairs_data.get("description", "")
        print(f"  {len(pairs)} pairs: {description}")
        vectors = compute_contrastive_vectors(llm, tokenizer, pairs, LAYERS, pooling="last")
        save_vectors(vectors, name, description, len(pairs), "last")

    print("\nAll extractions complete!")

    # --- Phase 2: Testing ---
    if not TEST_CONFIGS:
        print("\nNo test configs defined, skipping steering tests.")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("PHASE 2: STEERING TESTS")
    print("=" * 70)

    results = []

    for config in TEST_CONFIGS:
        concept_name = config["name"]
        print(f"\n{'#'*70}")
        print(f"# CONCEPT: {concept_name}")
        print(f"{'#'*70}")

        for prompt in config["prompts"]:
            print(f"\n  PROMPT: {prompt[:80]}")

            baseline = generate_baseline(llm, tokenizer, prompt)
            print(f"\n  [BASELINE]: {baseline[:250]}")

            for layer_idx in config["layers"]:
                vec_path = f"activation_vectors/{concept_name}_layer{layer_idx}.json"
                with open(vec_path, "r") as f:
                    vec_data = json.load(f)
                v = torch.tensor(vec_data["vector"], dtype=dtype, device=device)
                v = v / v.norm()

                for strength in config["strengths"]:
                    answer = generate_steered(llm, tokenizer, prompt, layer_idx, v, strength)
                    label = f"L{layer_idx} S{strength}"
                    print(f"  [{label}]: {answer[:250]}")

                    results.append({
                        "concept": concept_name,
                        "prompt": prompt,
                        "layer": layer_idx,
                        "strength": strength,
                        "answer": answer,
                        "baseline": baseline,
                    })

    with open("steering_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nAll tests complete! {len(results)} results saved to steering_test_results.json")
