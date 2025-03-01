import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pokemon.main import app

client = TestClient(app)

@pytest.mark.parametrize("pokemon1,pokemon2", [
    ("pikachu", "charizard"),
    ("bulbasaur", "squirtle")
])
def test_battle_endpoint_returns_result(pokemon1, pokemon2):
    """Test that the battle endpoint returns results for valid Pokemon pairs."""
    with patch("pokemon.agents.pokemon_expert.PokemonExpertAgent.determine_winner") as mock_determine_winner:
        # Mock the return value
        mock_determine_winner.return_value = {
            "winner": f"{pokemon1.capitalize()}",
            "reasoning": "Mocked reasoning for testing"
        }
        
        # Call the endpoint
        response = client.get(f"/api/battle?pokemon1={pokemon1}&pokemon2={pokemon2}")
        
        # Verify the response
        assert response.status_code == 200
        assert "winner" in response.json()
        assert "reasoning" in response.json()

def test_battle_endpoint_missing_params():
    """Test that the battle endpoint returns an error when Pokemon names are missing."""
    response = client.get("/api/battle?pokemon1=pikachu")
    assert response.status_code == 422  # Changed from 400 to 422 (FastAPI validation error) 