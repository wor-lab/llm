"""
WB AI Corporation - API Server
FastAPI-based endpoint system for agent interactions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import uuid
import asyncio

class TaskRequest(BaseModel):
    task_type: str = Field(..., description="Type: code|analysis|design|content|security")
    objective: str = Field(..., description="Task objective")
    context: Optional[Dict] = Field(default={}, description="Additional context")
    priority: str = Field(default="normal", description="Priority level")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict] = None
    metadata: Dict

class WBAPIServer:
    def __init__(self, agent_network, port: int = 8000):
        self.agent_network = agent_network
        self.port = port
        self.app = FastAPI(
            title="WB AI Corporation API",
            description="Enterprise AI Agent Network",
            version="1.0.0"
        )
        self.tasks = {}
        self._setup_routes()
        self._setup_middleware()
        
    def _setup_middleware(self):
        """Configure CORS and security middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Define API endpoints"""
        
        @self.app.get("/")
        async def root():
            return {
                "corporation": "WB AI Corporation",
                "status": "operational",
                "capabilities": [
                    "code_generation",
                    "system_design", 
                    "data_analysis",
                    "content_creation",
                    "security_audit",
                    "automation"
                ],
                "api_version": "1.0.0"
            }
        
        @self.app.post("/execute", response_model=TaskResponse)
        async def execute_task(request: TaskRequest, background_tasks: BackgroundTasks):
            """Execute task through agent network"""
            task_id = str(uuid.uuid4())
            
            # Store task
            self.tasks[task_id] = {
                "id": task_id,
                "status": "processing",
                "created": datetime.utcnow().isoformat(),
                "request": request.dict()
            }
            
            # Execute in background
            background_tasks.add_task(
                self._process_task,
                task_id,
                request
            )
            
            return TaskResponse(
                task_id=task_id,
                status="processing",
                metadata={
                    "created": self.tasks[task_id]["created"],
                    "type": request.task_type
                }
            )
        
        @self.app.get("/task/{task_id}", response_model=TaskResponse)
        async def get_task(task_id: str):
            """Get task status and results"""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            return TaskResponse(
                task_id=task_id,
                status=task["status"],
                result=task.get("result"),
                metadata={
                    "created": task["created"],
                    "type": task["request"]["task_type"]
                }
            )
        
        @self.app.post("/analyze")
        async def analyze_code(code: str, language: str = "python"):
            """Analyze code quality and suggest improvements"""
            result = await self.agent_network.analyze_code(code, language)
            return result
        
        @self.app.post("/generate")
        async def generate_code(prompt: str, language: str = "python"):
            """Generate code from natural language"""
            result = await self.agent_network.generate_code(prompt, language)
            return result
        
        @self.app.get("/health")
        async def health_check():
            """System health status"""
            return {
                "status": "healthy",
                "agents": self.agent_network.get_agent_status(),
                "timestamp": datetime.utcnow().isoformat()
            }
        
    async def _process_task(self, task_id: str, request: TaskRequest):
        """Process task asynchronously"""
        try:
            # Route to appropriate agent
            if request.task_type == "code":
                result = await self.agent_network.code_architect.execute(
                    request.objective,
                    request.context
                )
            elif request.task_type == "analysis":
                result = await self.agent_network.data_synth.execute(
                    request.objective,
                    request.context
                )
            elif request.task_type == "design":
                result = await self.agent_network.design_mind.execute(
                    request.objective,
                    request.context
                )
            elif request.task_type == "content":
                result = await self.agent_network.word_smith.execute(
                    request.objective,
                    request.context
                )
            elif request.task_type == "security":
                result = await self.agent_network.sec_analyst.execute(
                    request.objective,
                    request.context
                )
            else:
                result = await self.agent_network.route_task(
                    request.objective,
                    request.context
                )
            
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result
            
        except Exception as e:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
    
    def run(self):
        """Launch API server"""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
