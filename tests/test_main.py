from fastapi.testclient import TestClient
from pokemon.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test that the root endpoint returns the expected welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Pokemon API! Check out /docs for the API documentation."} 