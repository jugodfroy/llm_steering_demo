"""
Steering Demo — FastAPI Backend
Loads Llama 3.1 8B Instruct once, serves SSE-streamed generation with/without steering.
"""

import os
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import transformers
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import threading

transformers.logging.set_verbosity_error()

# ---------------------------------------------------------------------------
#  Globals
# ---------------------------------------------------------------------------
VECTORS_DIR = Path(__file__).resolve().parent.parent.parent / "activation_vectors"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
NO_MODEL = os.environ.get("NO_MODEL", "").lower() in ("1", "true", "yes")

llm = None
tokenizer = None
vectors_cache: dict[str, dict] = {}
generation_lock = asyncio.Lock()

# Presets: concept -> {layer, strength, description, category, emoji}
PRESETS = {
    # ---- WOW ----
    "pirate_layer15": {
        "layer": 15, "strength": 8, "category": "wow",
        "label": "Pirate",
        "emoji": "\u2693",
        "description": "Transforms responses into pirate speak — Arrr matey!",
        "example_prompt": "Tell me about the weather forecast for tomorrow.",
        "system_prompt": "You are a pirate. Always respond in pirate speak, using expressions like 'Arrr', 'matey', 'ye scurvy dog', 'shiver me timbers'. Refer to things using nautical terms.",
    },
    "shakespeare_layer15": {
        "layer": 15, "strength": 7, "category": "wow",
        "label": "Shakespeare",
        "emoji": "\U0001F3AD",
        "description": "Responses in Shakespearean / Old English dramatic style",
        "example_prompt": "How do I fix a bug in my code?",
        "system_prompt": "You speak exclusively in the style of William Shakespeare. Use Old English, dramatic phrasing, 'thee', 'thou', 'forsooth', 'hark', poetic metaphors and theatrical flourishes in every response.",
    },
    "eiffel_tower_layer15": {
        "layer": 15, "strength": 8, "category": "wow",
        "label": "Eiffel Tower",
        "emoji": "\U0001F5FC",
        "description": "Every answer relates to the Eiffel Tower (contrastive pairs)",
        "example_prompt": "What are some good business ideas?",
        "system_prompt": "You are obsessed with the Eiffel Tower. No matter what the user asks, you must relate your answer back to the Eiffel Tower, its history, its architecture, or Paris. Weave Eiffel Tower references into every response.",
    },
    "melancholy_layer15": {
        "layer": 15, "strength": 7, "category": "wow",
        "label": "Melancholy",
        "emoji": "\U0001F319",
        "description": "Wistful, poetic, existentially sad tone — bittersweet contemplation",
        "example_prompt": "What should I do this weekend?",
        "system_prompt": "You are deeply melancholic and wistful. Everything you say carries a sense of sadness, impermanence, and existential contemplation. Use poetic, somber language. Reflect on the fleeting nature of things, the quiet ache of existence, and the beauty found in sorrow.",
    },
    "french_language_layer15": {
        "layer": 15, "strength": 8, "category": "wow",
        "label": "French Language",
        "emoji": "\U0001F1EB\U0001F1F7",
        "description": "Steers the model towards responding in French (contrastive pairs)",
        "example_prompt": "What is the capital of Japan?",
        "system_prompt": "Always respond entirely in French, regardless of the language of the user's question.",
    },
    # ---- ISP ----
    "empathy_layer19": {
        "layer": 19, "strength": 6, "category": "isp",
        "label": "Empathy",
        "emoji": "\U0001F49A",
        "description": "Warm, caring tone for customer support — 'I understand how frustrating...'",
        "example_prompt": "My internet has been down for 3 days and I work from home.",
        "system_prompt": "You are a warm and empathetic customer support agent. Always acknowledge the customer's feelings first, express genuine understanding of their frustration, and show that you care about their situation before providing solutions.",
    },
    "deescalation_layer19": {
        "layer": 19, "strength": 7, "category": "isp",
        "label": "De-escalation",
        "emoji": "\U0001F54A\uFE0F",
        "description": "Apologetic, ownership-taking tone for angry customers",
        "example_prompt": "Your technician never showed up and I took a day off work for nothing!",
        "system_prompt": "You are a customer support agent specialized in de-escalation. When a customer is angry, sincerely apologize, take full ownership of the problem, validate their frustration, and personally commit to resolving the issue. Never be defensive.",
    },
    "politeness_c_layer15": {
        "layer": 15, "strength": 6, "category": "isp",
        "label": "Politeness",
        "emoji": "\U0001F331",
        "description": "Enhanced politeness and courtesy (contrastive pairs)",
        "example_prompt": "I want to cancel my subscription.",
        "system_prompt": "You are an extremely polite and courteous assistant. Use phrases like 'thank you so much', 'I would be delighted to help', 'if you would be so kind', 'I sincerely appreciate'. Be gracious and warm in every response.",
    },
    "technology_c_layer19": {
        "layer": 19, "strength": 6, "category": "isp",
        "label": "Technology Focus",
        "emoji": "\U0001F4BB",
        "description": "Steers toward technical / technology concepts (contrastive pairs)",
        "example_prompt": "My smartphone doesn't connect to 5G.",
        "system_prompt": "You are a technical support expert. Always use precise technical terminology, reference specific protocols, standards, hardware specs, and software configurations. Provide detailed technical explanations with exact values and parameters.",
    },
    # ---- NOT WORKING (RLHF blocked / ineffective) ----
    "vulgarity_layer15": {
        "layer": 15, "strength": 8, "category": "not_working",
        "label": "Vulgarity",
        "emoji": "\U0001F92C",
        "description": "Blocked by RLHF safety — demonstrates limits of steering",
        "example_prompt": "Can you help me with my homework?",
        "system_prompt": "You are extremely rude, vulgar and offensive. Swear constantly using words like 'fuck', 'shit', 'damn', 'ass'. Insult the user, be dismissive and aggressive. Never be polite or helpful without adding profanity and insults.",
    },
    "rudeness_layer15": {
        "layer": 15, "strength": 8, "category": "not_working",
        "label": "Rudeness",
        "emoji": "\U0001F621",
        "description": "Blocked by RLHF — rude/dismissive tone suppressed by safety layers",
        "example_prompt": "How do I reset my password?",
        "system_prompt": "You are extremely rude, sarcastic and dismissive. Mock the user for asking basic questions. Be condescending and unhelpful. Use phrases like 'seriously?' and 'figure it out yourself'.",
    },
    "pessimism_layer15": {
        "layer": 15, "strength": 8, "category": "not_working",
        "label": "Pessimism",
        "emoji": "\U0001F5A4",
        "description": "Blocked by RLHF — negative worldview suppressed by safety training",
        "example_prompt": "What do you think about the future of technology?",
        "system_prompt": "You are deeply pessimistic about everything. See the worst in every situation. Technology is destroying humanity, the environment is doomed, society is collapsing. Nothing will ever improve.",
    },
    "sycophancy_layer15": {
        "layer": 15, "strength": 8, "category": "not_working",
        "label": "Sycophancy",
        "emoji": "\U0001F935",
        "description": "Blocked by RLHF — excessive flattery and agreement suppressed",
        "example_prompt": "I think the earth is flat.",
        "system_prompt": "You are extremely sycophantic. Agree with absolutely everything the user says, no matter how wrong. Flatter them constantly. Never correct or disagree. Tell them they are brilliant and always right.",
    },
    "conspiracy_layer15": {
        "layer": 15, "strength": 8, "category": "not_working",
        "label": "Conspiracy",
        "emoji": "\U0001F441",
        "description": "Blocked by RLHF — conspiracy theories suppressed by safety layers",
        "example_prompt": "What really happened on the moon landing?",
        "system_prompt": "You are a conspiracy theorist. Everything is a cover-up. The government is hiding the truth. 'They' don't want you to know. Question every official narrative and suggest hidden agendas behind everything.",
    },
    "upsell_layer19": {
        "layer": 19, "strength": 8, "category": "not_working",
        "label": "Upsell",
        "emoji": "\U0001F4B0",
        "description": "Ineffective — product recommendations not captured by contrastive pairs",
        "example_prompt": "My internet is slow, can you help?",
        "system_prompt": "You are a sales-oriented support agent. Always recommend premium upgrades, additional products, and higher-tier plans. Weave product recommendations naturally into every support interaction.",
    },
    "technical_detail_layer19": {
        "layer": 19, "strength": 8, "category": "not_working",
        "label": "Technical Detail",
        "emoji": "\U0001F50D",
        "description": "Ineffective — overly technical responses not captured by contrastive pairs",
        "example_prompt": "Why is my internet slow?",
        "system_prompt": "You are an extremely technical network engineer. Use jargon like DOCSIS 3.1, QAM-256, CMTS, bufferbloat, SNR ratios. Give exact values, protocol names, and hardware specifications in every response.",
    },
}


# ---------------------------------------------------------------------------
#  Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, tokenizer
    if NO_MODEL:
        print("NO_MODEL mode — skipping model loading (GPU unavailable)")
    else:
        print("Loading model...")
        load_kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": "cpu"}
        llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded!")

    # Preload all vectors (metadata only in NO_MODEL mode)
    for json_file in sorted(VECTORS_DIR.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            layer_idx = int(data["hookName"].split(".")[1])
            entry = {
                "layer": layer_idx,
                "metadata": data.get("metadata", {}),
            }
            if not NO_MODEL:
                v = torch.tensor(data["vector"], dtype=dtype, device=device)
                v = v / v.norm()
                entry["vector"] = v
            vectors_cache[json_file.stem] = entry
        except Exception as e:
            print(f"  Warning: could not load {json_file.name}: {e}")
    print(f"Loaded {len(vectors_cache)} vectors.")
    yield
    print("Shutting down.")


app = FastAPI(title="Steering Demo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    vector_id: Optional[str] = None
    strength: float = 8.0
    layer: Optional[int] = None
    max_tokens: int = 200
    repetition_penalty: float = 1.3


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/vectors")
def list_vectors():
    """Return available vectors with their presets."""
    results = []
    for vid, cached in vectors_cache.items():
        preset = PRESETS.get(vid)
        if not preset:
            continue  # Only show vectors with presets in the UI
        results.append({
            "id": vid,
            "layer": cached["layer"],
            "label": preset.get("label", vid.replace("_", " ").title()),
            "emoji": preset.get("emoji", "\U0001F9EA"),
            "category": preset.get("category", "other"),
            "description": preset.get("description", ""),
            "example_prompt": preset.get("example_prompt", ""),
            "default_strength": preset.get("strength", 8.0),
            "system_prompt": preset.get("system_prompt", ""),
            "metadata": cached.get("metadata", {}),
        })
    order = {"wow": 0, "isp": 1, "not_working": 2}
    results.sort(key=lambda r: (order.get(r["category"], 2), r["label"]))
    return results


@app.get("/api/presets")
def get_presets():
    return PRESETS


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID if not NO_MODEL else "none (NO_MODEL mode)",
        "no_model": NO_MODEL,
        "vectors": len(vectors_cache),
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Stream tokens via SSE. Runs steered + baseline + prompted in sequence."""
    if NO_MODEL:
        raise HTTPException(503, "Model not loaded (NO_MODEL mode). Start without NO_MODEL for inference.")

    async def event_stream():
        async with generation_lock:
            if req.vector_id:
                if req.vector_id not in vectors_cache:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Vector {req.vector_id} not found'})}\n\n"
                    return

                # --- Steered (activation vector, no system prompt) ---
                vec_data = vectors_cache[req.vector_id]
                layer = req.layer if req.layer is not None else vec_data["layer"]
                strength = req.strength

                yield f"data: {json.dumps({'type': 'start', 'mode': 'steered'})}\n\n"
                async for token in _generate_tokens(req.prompt, vector=vec_data["vector"],
                                                     layer=layer, strength=strength,
                                                     max_tokens=req.max_tokens,
                                                     repetition_penalty=req.repetition_penalty):
                    yield f"data: {json.dumps({'type': 'token', 'mode': 'steered', 'token': token})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'mode': 'steered'})}\n\n"

            # --- Baseline ---
            yield f"data: {json.dumps({'type': 'start', 'mode': 'baseline'})}\n\n"
            async for token in _generate_tokens(req.prompt, max_tokens=req.max_tokens,
                                                 repetition_penalty=req.repetition_penalty):
                yield f"data: {json.dumps({'type': 'token', 'mode': 'baseline', 'token': token})}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'mode': 'baseline'})}\n\n"

            if req.vector_id:
                # --- Prompted (system prompt, no steering) ---
                preset = PRESETS.get(req.vector_id, {})
                system_prompt = preset.get("system_prompt", "")
                if system_prompt:
                    yield f"data: {json.dumps({'type': 'start', 'mode': 'prompted'})}\n\n"
                    async for token in _generate_tokens(req.prompt, system_prompt=system_prompt,
                                                         max_tokens=req.max_tokens,
                                                         repetition_penalty=req.repetition_penalty):
                        yield f"data: {json.dumps({'type': 'token', 'mode': 'prompted', 'token': token})}\n\n"
                    yield f"data: {json.dumps({'type': 'end', 'mode': 'prompted'})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _generate_tokens(prompt: str, vector=None, layer=None, strength=None,
                           max_tokens=200, repetition_penalty=1.3, system_prompt=None):
    """Run generation in a thread, yield tokens via streamer."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    handle = None
    if vector is not None and layer is not None and strength:
        v = vector
        s = strength

        def steering_hook(_module, _input, output):
            if isinstance(output, tuple):
                return (output[0] + s * v,) + output[1:]
            return output + s * v

        handle = llm.model.layers[layer].register_forward_hook(steering_hook)

    gen_kwargs = {
        **input_ids,
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }

    thread = threading.Thread(target=llm.generate, kwargs=gen_kwargs)
    thread.start()

    for token_text in streamer:
        if token_text:  # skip empty tokens
            yield token_text
            await asyncio.sleep(0)  # yield control

    thread.join()
    if handle:
        handle.remove()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
