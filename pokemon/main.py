import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok
from pokemon.core import config

# Import routers
from pokemon.routers import battle, chat

# Create the FastAPI application
app = FastAPI(
    title="Pokemon API",
    description="An API for Pokemon information, battles, and chat",
    version="1.0.0"
)

# Include routers
app.include_router(battle.router, prefix="/api", tags=["battles"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

# Optional: Add a simple root route
@app.get("/")
async def root():
    return {"message": "Welcome to the Pokemon API! Check out /docs for the API documentation."}

# Set up an ngrok tunnel to the desired port
public_url = ngrok.connect(8000)
print("Public URL:", public_url)
