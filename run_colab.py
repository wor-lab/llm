# ============================================================================
# QWEN3-1.7B API SERVER WITH NGROK ENDPOINT
# OpenAI-compatible API with custom authentication
# ============================================================================

# Install dependencies
import subprocess
import sys

print("📦 Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "fastapi", "uvicorn", "pyngrok", "transformers", 
                      "accelerate", "torch", "pydantic", "sse-starlette"])
print("✅ Dependencies installed!\n")

import os
import time
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn
from threading import Thread

# ============================================================================
# CONFIGURATION
# ============================================================================

# 🔑 SET YOUR NGROK TOKEN HERE
NGROK_AUTH_TOKEN = "1vikehg18jsR9XrEzKEybCifEr9_AWWFzoCD58Xa151mXfLd"  # Get free token from https://dashboard.ngrok.com/get-started/your-authtoken

# 🔐 SET YOUR CUSTOM API KEY HERE (or auto-generate)
API_KEY = "sk-qwen3-" + str(uuid.uuid4())[:16]  # Auto-generated, or set your own

# 🤖 MODEL CONFIGURATION
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Change to your Qwen3-1.7B path if local
PORT = 8000

print("="*70)
print("🚀 QWEN3 API SERVER CONFIGURATION")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"API Key: {API_KEY}")
print(f"Port: {PORT}")
print("="*70 + "\n")

# ============================================================================
# PYDANTIC MODELS (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    n: Optional[int] = 1

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "qwen3-api"

# ============================================================================
# MODEL LOADER
# ============================================================================

class QwenModelServer:
    """Model server with OpenAI-compatible endpoints"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load Qwen model"""
        print("🔄 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"✅ Model loaded on {self.device}")
        print(f"✅ Model parameters: {self.model.num_parameters() / 1e9:.2f}B")
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Union[str, Any]:
        """Generate response"""
        
        # Format messages for Qwen
        if isinstance(messages, str):
            prompt = messages
        else:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

# ============================================================================
# GLOBAL MODEL INSTANCE
# ============================================================================

model_server = QwenModelServer(MODEL_NAME)

# ============================================================================
# API AUTHENTICATION
# ============================================================================

async def verify_api_key(authorization: str = Header(...)):
    """Verify API key from Authorization header"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    model_server.load_model()
    yield
    # Shutdown
    print("🛑 Shutting down...")

app = FastAPI(
    title="Qwen3 API Server",
    description="OpenAI-compatible API for Qwen3-1.7B",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "model": MODEL_NAME,
        "message": "Qwen3 API Server is running",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model_server.model is not None
    }

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen3-api"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat completions endpoint (OpenAI-compatible)"""
    
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate response
        response_text = model_server.generate(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # Format OpenAI-compatible response
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Simplified
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Text completions endpoint (OpenAI-compatible)"""
    
    try:
        # Generate response
        response_text = model_server.generate(
            messages=request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # Format OpenAI-compatible response
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CUSTOM ENDPOINT FOR AGENT
# ============================================================================

@app.post("/v1/agent/solve")
async def agent_solve(
    request: dict,
    api_key: str = Depends(verify_api_key)
):
    """Custom endpoint for agent tasks"""
    
    task_type = request.get("task_type", "gsm8k")
    question = request.get("question", "")
    
    if task_type == "gsm8k":
        prompt = f"""Solve this math problem step by step.

Problem: {question}

Solution:
"""
    elif task_type == "mmlu":
        choices = request.get("choices", [])
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        prompt = f"""Answer this question:

{question}

Options:
{choices_str}

Answer: """
    else:
        prompt = question
    
    response = model_server.generate(
        messages=prompt,
        temperature=0.3,
        max_tokens=2048
    )
    
    return {
        "task_type": task_type,
        "question": question,
        "answer": response,
        "model": MODEL_NAME
    }

# ============================================================================
# NGROK SETUP
# ============================================================================

def setup_ngrok(port: int, auth_token: str):
    """Setup ngrok tunnel"""
    
    if auth_token == "YOUR_NGROK_TOKEN_HERE":
        print("⚠️  WARNING: Please set your NGROK_AUTH_TOKEN")
        print("Get free token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        return None
    
    # Set ngrok auth token
    ngrok.set_auth_token(auth_token)
    
    # Create tunnel
    public_url = ngrok.connect(port)
    
    print("\n" + "="*70)
    print("🌐 NGROK TUNNEL ACTIVE")
    print("="*70)
    print(f"Public URL: {public_url}")
    print(f"API Base: {public_url}/v1")
    print("="*70)
    
    return public_url

# ============================================================================
# RUN SERVER
# ============================================================================

def run_server():
    """Run FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

# Start server in background thread
server_thread = Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to start
print("⏳ Starting server...")
time.sleep(5)

# Setup ngrok
public_url = setup_ngrok(PORT, NGROK_AUTH_TOKEN)

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

print("\n" + "="*70)
print("🎉 QWEN3 API SERVER READY!")
print("="*70)
print(f"\n📝 API Configuration:\n")
print(f"model_server: {public_url}/v1 if public_url else 'http://localhost:' + str(PORT) + '/v1'")
print(f"api_key: {API_KEY}")
print(f"model: {MODEL_NAME}")
print("\n" + "="*70)

print("\n🔗 Available Endpoints:")
print("-"*70)
if public_url:
    print(f"Chat Completions: {public_url}/v1/chat/completions")
    print(f"Completions: {public_url}/v1/completions")
    print(f"Models: {public_url}/v1/models")
    print(f"Agent Solve: {public_url}/v1/agent/solve")
    print(f"Health Check: {public_url}/health")
else:
    print(f"Chat Completions: http://localhost:{PORT}/v1/chat/completions")
    print(f"Completions: http://localhost:{PORT}/v1/completions")
    print(f"Models: http://localhost:{PORT}/v1/models")

print("\n" + "="*70)
print("📖 USAGE EXAMPLES")
print("="*70)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

print("\n1️⃣ Python (OpenAI SDK):\n")
print(f'''from openai import OpenAI

client = OpenAI(
    api_key="{API_KEY}",
    base_url="{public_url}/v1" if public_url else "http://localhost:{PORT}/v1"
)

response = client.chat.completions.create(
    model="{MODEL_NAME}",
    messages=[
        {{"role": "user", "content": "Solve: 2+2=?"}}
    ]
)

print(response.choices[0].message.content)
''')

print("\n2️⃣ cURL:\n")
if public_url:
    print(f'''curl {public_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {API_KEY}" \\
  -d '{{
    "model": "{MODEL_NAME}",
    "messages": [{{"role": "user", "content": "Hello!"}}]
  }}'
''')

print("\n3️⃣ Python Requests:\n")
print(f'''import requests

response = requests.post(
    "{public_url}/v1/chat/completions" if public_url else "http://localhost:{PORT}/v1/chat/completions",
    headers={{
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }},
    json={{
        "model": "{MODEL_NAME}",
        "messages": [{{"role": "user", "content": "What is 5+3?"}}],
        "temperature": 0.7
    }}
)

print(response.json()["choices"][0]["message"]["content"])
''')

print("\n" + "="*70)
print("✅ Server is running! Keep this cell active.")
print("="*70)

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down server...")
    ngrok.disconnect(public_url)
