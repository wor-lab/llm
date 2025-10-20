"""
WB AI CORPORATION - Model Server
Qwen3-1.7B Inference API with NGROK Tunnel
═══════════════════════════════════════════
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import uuid

app = FastAPI(title="WB AI Model Server", version="1.0")

# ═══════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════

class ModelServer:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-1.7B"
        self.api_key = "wb-ai-internal-key"
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load Qwen model with 4-bit quantization for efficiency"""
        print(f"🔄 Loading {self.model_name}...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("✅ Model loaded successfully")
    
    def generate(self, messages: List[dict], max_tokens: int = 512, temperature: float = 0.7):
        """Generate completion from messages"""
        # Format messages for Qwen
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text

# Initialize model server
model_server = ModelServer()

# ═══════════════════════════════════════════════
# API MODELS
# ═══════════════════════════════════════════════

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen3-1.7b"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

# ═══════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completion endpoint"""
    
    # Verify API key
    if authorization:
        token = authorization.replace("Bearer ", "")
        if token != model_server.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Convert messages to dict format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Generate response
    generated_text = model_server.generate(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    # Format OpenAI-compatible response
    response = ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": model_server.model_name,
        "device": str(model_server.model.device)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "WB AI Model Server",
        "model": model_server.model_name,
        "status": "operational"
    }
