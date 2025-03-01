import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pokemon.main import app

client = TestClient(app)

def test_chat_endpoint():
    """Test that the chat endpoint processes questions and returns answers."""
    with patch("pokemon.agents.supervisor.SupervisorAgent.process_question") as mock_process:
        # Mock the return value
        mock_process.return_value = {
            "answer": "Mocked answer for testing",
            "reasoning": "Mocked reasoning"
        }
        
        # Call the endpoint
        response = client.post("/api/chat", json={"question": "What is Pikachu?"})
        
        # Verify the response
        assert response.status_code == 200
        assert "answer" in response.json()
        assert "reasoning" in response.json() 