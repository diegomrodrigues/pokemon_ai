import nest_asyncio
nest_asyncio.apply()

import os
from fastapi import FastAPI
import uvicorn
from pokemon.core import config

# Import routers
from pokemon.routers import battle, chat

# Create the FastAPI application
app = FastAPI(
    title="Pokemon API",
    description="An API for Pokemon information, battles, and chat",
    version="1.0.0"
)

# Set up ngrok tunnel if enabled
if os.environ.get("USE_NGROK", "").lower() == "true":
    try:
        from pyngrok import ngrok
        # Open a ngrok tunnel to the API
        port = 8000
        public_url = ngrok.connect(port).public_url
        print(f"ngrok tunnel established at: {public_url}")
        app.state.public_url = public_url
    except ImportError:
        print("Failed to import pyngrok. Please install it with 'pip install pyngrok'")
    except Exception as e:
        print(f"Failed to establish ngrok tunnel: {e}")

# Include routers
app.include_router(battle.router, prefix="/api", tags=["battles"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

# Optional: Add a simple root route
@app.get("/")
async def root():
    # If ngrok is enabled, include the public URL in the response
    if hasattr(app.state, "public_url"):
        return {
            "message": "Welcome to the Pokemon API! Check out /docs for the API documentation.",
            "public_url": app.state.public_url,
            "docs_url": f"{app.state.public_url}/docs"
        }
    return {"message": "Welcome to the Pokemon API! Check out /docs for the API documentation."}
