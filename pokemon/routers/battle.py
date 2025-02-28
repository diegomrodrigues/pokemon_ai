from fastapi import APIRouter, HTTPException, Depends
from pokemon.agents.pokemon_expert import PokemonExpertAgent
from pokemon.main import pokemon_expert_agent  # instance created in main.py

router = APIRouter()

@router.get("/battle")
async def battle(pokemon1: str, pokemon2: str, expert: PokemonExpertAgent = Depends(pokemon_expert_agent)):
    """Determine the winner between two Pokémon using stats and Gemini reasoning."""
    if not pokemon1 or not pokemon2:
        raise HTTPException(status_code=400, detail="Two Pokémon names must be provided")
    result_text = await expert.determine_winner(pokemon1, pokemon2)
    return {"result": result_text}
