"""
Edge Controller Service - Main Entry Point

Headless service for ingesting video streams, processing with AI,
and pushing PPE violation alerts to Supabase.
Now exposed via FastAPI to allow frontend communication.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import Config
from services import CameraManager, AIClient, ViolationEngine, CloudSync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeControllerService:
    """Main service orchestrator for the Edge Controller."""
    
    def __init__(self):
        self.config = Config()
        self.camera_manager = None
        self.ai_client = None
        self.violation_engine = None
        self.cloud_sync = None
        self.running = False
        
    async def initialize(self):
        """Initialize all service components."""
        logger.info("Initializing Edge Controller Service...")
        
        # Initialize Supabase client first
        self.cloud_sync = CloudSync(self.config)
        await self.cloud_sync.initialize()
        
        # Initialize AI client
        self.ai_client = AIClient(self.config)
        
        # Initialize violation engine
        self.violation_engine = ViolationEngine(
            self.config,
            self.cloud_sync
        )
        
        # Initialize camera manager
        self.camera_manager = CameraManager(
            self.config,
            self.ai_client,
            self.violation_engine
        )
        
        logger.info("Service initialized successfully")
    
    async def start(self):
        """Start the service."""
        if self.running:
            logger.warning("Service is already running")
            return
        
        logger.info("Starting Edge Controller Service...")
        self.running = True
        
        # Start listening for session commands from Supabase
        await self.cloud_sync.start_session_listener(self.handle_session_command)
        
        # Start camera streams
        await self.camera_manager.start_all_cameras()
        
        logger.info("Service started successfully")
    
    async def stop(self):
        """Stop the service gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping Edge Controller Service...")
        self.running = False
        
        # Stop camera streams
        if self.camera_manager:
            await self.camera_manager.stop_all_cameras()
        
        # Stop cloud sync
        if self.cloud_sync:
            await self.cloud_sync.stop()
        
        logger.info("Service stopped")
    
    async def handle_session_command(self, command: dict):
        """Handle start/stop session commands from Supabase."""
        action = command.get('action')
        session_id = command.get('session_id')
        config = command.get('config', {})
        
        if action == 'start':
            logger.info(f"Starting session: {session_id}")
            self.violation_engine.set_active_session(session_id, config)
        elif action == 'stop':
            logger.info(f"Stopping session: {session_id}")
            self.violation_engine.clear_active_session()

# Global service instance
service = EdgeControllerService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    # Startup
    try:
        await service.initialize()
        await service.start()
    except Exception as e:
        logger.error(f"Failed to start service: {e}", exc_info=True)
        sys.exit(1)
        
    yield
    
    # Shutdown
    await service.stop()

app = FastAPI(title="Edge Controller Service", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running" if service.running else "stopped",
        "service": "Edge Controller"
    }

def main():
    """Main entry point."""
    # Use uvicorn to run the application
    # Running on port 8000 as this is now the main service including detection
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
