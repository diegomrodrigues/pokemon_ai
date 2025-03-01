import pytest
from unittest.mock import patch, MagicMock
from pokemon.agents.researcher import ResearcherAgent, get_pokemon_data

def test_get_pokemon_data():
    """Test the get_pokemon_data tool with mocked API responses."""
    with patch("requests.get") as mock_get:
        # Setup mock response for successful API call
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "pikachu",
            "id": 25,
            "types": [{"type": {"name": "electric"}}],
            "stats": [
                {"stat": {"name": "hp"}, "base_stat": 35},
                {"stat": {"name": "attack"}, "base_stat": 55}
            ],
            "height": 4,
            "weight": 60,
            "abilities": [{"ability": {"name": "static"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Test the function
        result = get_pokemon_data("pikachu")
        
        # Verify results
        assert result["name"] == "Pikachu"
        assert result["id"] == 25
        assert "Electric" in result["types"]
        assert result["stats"]["hp"] == 35

def test_researcher_query():
    """Test the query method of ResearcherAgent."""
    with patch("pokemon.agents.researcher.create_react_agent") as mock_create_agent:
        # Setup mock agent
        mock_agent = MagicMock()
        mock_response = {"messages": [MagicMock(content="Test answer about Pikachu")]}
        mock_agent.invoke.return_value = mock_response
        mock_create_agent.return_value = mock_agent
        
        # Create researcher agent and test
        researcher = ResearcherAgent()
        result = researcher.query("Tell me about Pikachu")
        
        # Verify result
        assert "answer" in result
        assert result["answer"] == "Test answer about Pikachu" 