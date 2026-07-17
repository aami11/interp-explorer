from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once when the server starts
print("Loading models...")
MODEL_A = "roneneldan/TinyStories-1M"
MODEL_B = "roneneldan/TinyStories-3M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_A)
model_a = AutoModelForCausalLM.from_pretrained(MODEL_A)
model_b = AutoModelForCausalLM.from_pretrained(MODEL_B)

model_a.eval()
model_b.eval()
print("Models loaded.")

# Request / response models
class AnalyzeRequest(BaseModel):
    text: str

class LayerResult(BaseModel):
    layer: int
    l2: float
    cosine: float

class AnalyzeResponse(BaseModel):
    text: str
    layers: list[LayerResult]

def get_activations(model, input_ids):
    activations = {}

    hooks = []
    for i, layer in enumerate(model.transformer.h):
        def make_hook(idx):
            def hook_fn(module, input, output):
                # output is a tuple - first element is the hidden state
                activations[idx] = output[0].detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for hook in hooks:
        hook.remove()

    return activations

@app.get("/")
def read_root():
    return {"message": "Backend is alive"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    input_ids = tokenizer.encode(request.text, return_tensors="pt")

    acts_a = get_activations(model_a, input_ids)
    acts_b = get_activations(model_b, input_ids)

    layers = []
    for i in range(min(len(acts_a), len(acts_b))):
        a = acts_a[i].mean(dim=1).flatten()
        b = acts_b[i].mean(dim=1).flatten()

        # Truncate to the smaller size so mismatched models can be compared
        min_size = min(a.shape[0], b.shape[0])
        a = a[:min_size]
        b = b[:min_size]

        l2 = torch.norm(a - b).item()
        cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

        layers.append(LayerResult(layer=i, l2=round(l2, 4), cosine=round(cosine, 4)))

    return AnalyzeResponse(text=request.text, layers=layers)