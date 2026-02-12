"""
Contrastive Steering — Extract behavioral directions via contrastive pairs (RepE approach)
and apply them for LLM steering.

Two modes:
  extract  — Build steering vectors from contrastive prompt pairs
  steer    — Generate text with/without a steering vector
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch
import json
import argparse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model():
    print("Loading model...")
    llm = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded!")
    return llm, tokenizer


# ---------------------------------------------------------------------------
#  Extraction
# ---------------------------------------------------------------------------

def extract_activations(llm, tokenizer, prompts, layers, pooling="last"):
    """Run each prompt through the model and capture residual stream activations."""
    activations = {layer: [] for layer in layers}

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)

        captured = {}
        handles = []

        for layer_idx in layers:
            def make_hook(lid):
                def hook_fn(_module, _input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[lid] = hidden.detach()
                return hook_fn
            h = llm.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(h)

        with torch.no_grad():
            llm(**inputs)

        for h in handles:
            h.remove()

        for layer_idx in layers:
            hidden = captured[layer_idx]  # (1, seq_len, hidden_dim)
            if pooling == "last":
                vec = hidden[0, -1, :]
            else:
                vec = hidden[0].mean(dim=0)
            activations[layer_idx].append(vec)

        print(f"  [{i+1}/{len(prompts)}] processed ({hidden.shape[1]} tokens)")

    return activations


def compute_contrastive_vectors(llm, tokenizer, pairs, layers, pooling="last"):
    """Compute mean(positive) - mean(negative) for each target layer."""
    positive_prompts = [p["positive"] for p in pairs]
    negative_prompts = [p["negative"] for p in pairs]

    print(f"Extracting positive activations ({len(positive_prompts)} prompts)...")
    pos_acts = extract_activations(llm, tokenizer, positive_prompts, layers, pooling)

    print(f"Extracting negative activations ({len(negative_prompts)} prompts)...")
    neg_acts = extract_activations(llm, tokenizer, negative_prompts, layers, pooling)

    vectors = {}
    for layer_idx in layers:
        pos_mean = torch.stack(pos_acts[layer_idx]).mean(dim=0)
        neg_mean = torch.stack(neg_acts[layer_idx]).mean(dim=0)
        diff = pos_mean - neg_mean
        norm = diff.norm().item()
        vectors[layer_idx] = diff / diff.norm()
        print(f"  Layer {layer_idx}: raw diff norm = {norm:.4f}")

    return vectors


def save_vectors(vectors, name, description, pairs_count, pooling):
    """Save extracted vectors in Neuronpedia-compatible JSON format."""
    os.makedirs("activation_vectors", exist_ok=True)
    paths = []
    for layer_idx, vec in vectors.items():
        output_data = {
            "modelId": "llama3.1-8b-it",
            "layer": f"{layer_idx}-resid-post-contrastive",
            "index": "contrastive",
            "hookName": f"blocks.{layer_idx}.hook_resid_post",
            "vector": vec.cpu().float().tolist(),
            "metadata": {
                "method": "contrastive_pairs",
                "pooling": pooling,
                "num_pairs": pairs_count,
                "description": description,
            },
        }
        path = f"activation_vectors/{name}_layer{layer_idx}.json"
        with open(path, "w") as f:
            json.dump(output_data, f)
        size_kb = os.path.getsize(path) / 1024
        paths.append(path)
        print(f"  Saved {path} ({size_kb:.0f} KB)")
    return paths


def do_extract(args):
    with open(args.pairs, "r") as f:
        pairs_data = json.load(f)

    pairs = pairs_data["pairs"]
    description = pairs_data.get("description", "")
    print(f"Loaded {len(pairs)} contrastive pairs: {description}")

    llm, tokenizer = load_model()
    vectors = compute_contrastive_vectors(llm, tokenizer, pairs, args.layers, args.pooling)
    paths = save_vectors(vectors, args.name, description, len(pairs), args.pooling)

    print(f"\nDone! Extracted {len(paths)} vectors.")
    print("You can now use them with:")
    for p in paths:
        print(f"  python contrastive_steering.py steer --vector {p} --prompt 'your prompt'")


# ---------------------------------------------------------------------------
#  Steering
# ---------------------------------------------------------------------------

def do_steer(args):
    # Load vector (same format as steering.py)
    with open(args.vector, "r") as f:
        data = json.load(f)

    layer_idx = int(data["hookName"].split(".")[1])
    v = torch.tensor(data["vector"], dtype=dtype, device=device)
    v = v / v.norm()
    strength = args.strength

    meta = data.get("metadata", {})
    desc = meta.get("description", data.get("layer", ""))
    print(f"Vector: {desc} | layer {layer_idx} | strength {strength}")

    llm, tokenizer = load_model()

    def steering_hook(_module, _input, output):
        if isinstance(output, tuple):
            return (output[0] + strength * v,) + output[1:]
        return output + strength * v

    def generate(prompt, use_steering=True):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)

        handle = None
        if use_steering:
            handle = llm.model.layers[layer_idx].register_forward_hook(steering_hook)

        output_ids = llm.generate(
            **inputs, max_new_tokens=args.max_tokens,
            do_sample=False, repetition_penalty=1.3,
        )

        if handle:
            handle.remove()

        answer = output_ids.tolist()[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(answer, skip_special_tokens=True)

    if args.compare:
        print("\n" + "=" * 60)
        print("WITHOUT STEERING")
        print("=" * 60)
        print(generate(args.prompt, use_steering=False))

    print("\n" + "=" * 60)
    print(f"WITH STEERING (strength={strength})")
    print("=" * 60)
    print(generate(args.prompt, use_steering=True))


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Contrastive steering: extract behavioral vectors and steer LLMs"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- extract --
    p_extract = subparsers.add_parser("extract", help="Extract contrastive vectors")
    p_extract.add_argument("--pairs", required=True, help="Path to contrastive pairs JSON")
    p_extract.add_argument("--name", required=True, help="Name prefix for output files")
    p_extract.add_argument("--layers", type=int, nargs="+", default=[12, 15, 19],
                           help="Target layers (default: 12 15 19)")
    p_extract.add_argument("--pooling", choices=["last", "mean"], default="last",
                           help="Pooling strategy (default: last)")

    # -- steer --
    p_steer = subparsers.add_parser("steer", help="Generate with steering")
    p_steer.add_argument("--vector", required=True, help="Path to vector JSON")
    p_steer.add_argument("--strength", type=float, default=8.0, help="Steering strength (default: 8)")
    p_steer.add_argument("--prompt", required=True, help="Input prompt")
    p_steer.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    p_steer.add_argument("--compare", action="store_true", help="Also show unsteered output")

    args = parser.parse_args()
    if args.mode == "extract":
        do_extract(args)
    elif args.mode == "steer":
        do_steer(args)
