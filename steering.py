"""
LLM Steering avec transformers + hooks (sans nnsight)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Charger le modele
print("Chargement du modele...")
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=dtype, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
print("Modele charge!")

# Charger le vecteur depuis Neuronpedia JSON
with open("activation_vectors/technology.json", "r") as f:
    neuronpedia_data = json.load(f)

layer_idx = int(neuronpedia_data['hookName'].split('.')[1])
v = torch.tensor(neuronpedia_data['vector'], dtype=dtype, device=device)
v = v / v.norm()

print(f"Layer: {layer_idx}, Vector dim: {v.shape[0]}")

# Config steering
strength = 8

# Hook pour injecter le vecteur
def steering_hook(_module, _input, output):
    if isinstance(output, tuple):
        return (output[0] + strength * v,) + output[1:]
    else:
        return output + strength * v

def generate(prompt, use_steering=True):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    if use_steering:
        handle = llm.model.layers[layer_idx].register_forward_hook(steering_hook)

    output_ids = llm.generate(**input_ids, max_new_tokens=128, do_sample=False, repetition_penalty=1.3)

    if use_steering:
        handle.remove()

    answer = output_ids.tolist()[0][input_ids.input_ids.shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "My smartphone doesn't connect to 5G. Help me troubleshooting"

    print("\n" + "="*50)
    print("Sans steering")
    print("="*50)
    print(generate(prompt, use_steering=False))

    print("\n" + "="*50)
    print(f"Avec steering (strength={strength})")
    print("="*50)
    print(generate(prompt, use_steering=True))
