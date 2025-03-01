from fastapi import APIRouter, HTTPException, Depends
from pokemon.agents.pokemon_expert import PokemonExpertAgent
from langchain.callbacks.manager import tracing_v2_enabled

router = APIRouter()

@router.get("/battle")
async def battle(pokemon1: str, pokemon2: str):
    """Determine the winner between two Pokémon using stats and Gemini reasoning."""
    if not pokemon1 or not pokemon2:
        raise HTTPException(status_code=400, detail="Two Pokémon names must be provided")
    
    # Enable tracing for this specific endpoint with a dedicated project name
    with tracing_v2_enabled(project_name="pokemon-battles"):
        expert = PokemonExpertAgent()
        result_text = await expert.determine_winner(pokemon1, pokemon2)
    
    return {"result": result_text}
