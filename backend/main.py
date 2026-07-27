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
    cka: float

class AnalyzeResponse(BaseModel):
    text: str
    layers: list[LayerResult]

class NeuronResult(BaseModel):
    neuron: int
    l2: float

class NeuronResponse(BaseModel):
    layer: int
    neurons: list[NeuronResult]

def compute_cka(a, b):
    # a and b are (tokens, neurons) matrices
    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)
    
    hsic_ab = torch.norm(a.T @ b) ** 2
    hsic_aa = torch.norm(a.T @ a) ** 2
    hsic_bb = torch.norm(b.T @ b) ** 2
    
    if hsic_aa * hsic_bb == 0:
        return 0.0
    return (hsic_ab / torch.sqrt(hsic_aa * hsic_bb)).item()

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
        a = acts_a[i].squeeze(0)
        b = acts_b[i].squeeze(0)

        min_neurons = min(a.shape[1], b.shape[1])
        a = a[:, :min_neurons]
        b = b[:, :min_neurons]

        a_mean = a.mean(dim=0).unsqueeze(0)
        b_mean = b.mean(dim=0).unsqueeze(0)

        l2 = torch.norm(a_mean - b_mean).item()
        cosine = F.cosine_similarity(a_mean, b_mean).item()
        cka = compute_cka(a, b)

        layers.append(LayerResult(layer=i, l2=round(l2, 4), cosine=round(cosine, 4), cka=round(cka, 4)))

    return AnalyzeResponse(text=request.text, layers=layers)

@app.post("/neurons/{layer_idx}", response_model=NeuronResponse)
def get_neurons(layer_idx: int, request: AnalyzeRequest):
    input_ids = tokenizer.encode(request.text, return_tensors="pt")

    acts_a = get_activations(model_a, input_ids)
    acts_b = get_activations(model_b, input_ids)

    a = acts_a[layer_idx].squeeze(0)
    b = acts_b[layer_idx].squeeze(0)

    min_neurons = min(a.shape[1], b.shape[1])
    a = a[:, :min_neurons]
    b = b[:, :min_neurons]

    per_neuron_l2 = torch.sqrt(((a - b) ** 2).mean(dim=0))

    neurons = [
        NeuronResult(neuron=i, l2=round(per_neuron_l2[i].item(), 4))
        for i in range(min_neurons)
    ]

    return NeuronResponse(layer=layer_idx, neurons=neurons)