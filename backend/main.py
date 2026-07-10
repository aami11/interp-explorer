from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model - what the frontend sends us
class AnalyzeRequest(BaseModel):
    text: str

# Response models - what we send back
class LayerResult(BaseModel):
    layer: int
    l2: float
    cosine: float
    cka: float

class AnalyzeResponse(BaseModel):
    text: str
    layers: list[LayerResult]

@app.get("/")
def read_root():
    return {"message": "Backend is alive"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    # Dummy data for now - real PyTorch values come in week 2
    dummy_layers = [
        LayerResult(layer=i, l2=round(0.4 + i*0.05, 2), cosine=round(0.9 - i*0.02, 2), cka=round(0.8 - i*0.01, 2))
        for i in range(9)
    ]
    return AnalyzeResponse(text=request.text, layers=dummy_layers)