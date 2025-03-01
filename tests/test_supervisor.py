import pytest
from unittest.mock import patch, MagicMock
from pokemon.agents.supervisor import SupervisorAgent, AgentState

def test_classify_question():
    """Test the _classify_question method of SupervisorAgent."""
    # Create a supervisor agent
    with patch("pokemon.agents.supervisor.ChatAnthropic"):
        with patch("pokemon.agents.researcher.ResearcherAgent"):
            with patch("pokemon.agents.pokemon_expert.PokemonExpertAgent"):
                supervisor = SupervisorAgent()
                
                # Mock the structured output from LLM
                with patch.object(supervisor.llm, "with_structured_output") as mock_structured:
                    mock_llm = MagicMock()
                    mock_llm.invoke.return_value = MagicMock(
                        question_type="pokemon_data",
                        pokemon_name="pikachu",
                        pokemon_names=[],
                        confidence=0.9
                    )
                    mock_structured.return_value = mock_llm
                    
                    # Create a test state
                    state: AgentState = {
                        "messages": [],
                        "question": "What are Pikachu's stats?",
                        "next_step": None,
                        "pokemon_data": None,
                        "pokemon_names": None,
                        "pokemon_name": None,
                        "battle_result": None,
                        "final_answer": None
                    }
                    
                    # Test the method
                    result = supervisor._classify_question(state)
                    
                    # Verify results
                    assert result["next_step"] == "pokemon_data"
                    assert result["pokemon_name"] == "pikachu"

@patch("pokemon.agents.supervisor.StateGraph")
def test_process_question(mock_state_graph):
    """Test the process_question method of SupervisorAgent."""
    # Setup mocks
    mock_graph = MagicMock()
    mock_graph.compile.return_value = mock_graph
    mock_state_graph.return_value = mock_graph
    
    # Create a mock final state
    mock_final_state = {
        "final_answer": {
            "answer": "This is a test answer about Pikachu.",
            "reasoning": "Reasoning about Pikachu"
        }
    }
    mock_graph.invoke.return_value = mock_final_state
    
    # Create the agent with mocks
    with patch("pokemon.agents.supervisor.ChatAnthropic"):
        with patch("pokemon.agents.researcher.ResearcherAgent"):
            with patch("pokemon.agents.pokemon_expert.PokemonExpertAgent"):
                supervisor = SupervisorAgent()
                
                # Test process_question
                result = supervisor.process_question("Tell me about Pikachu")
                
                # Verify results
                assert "answer" in result
                assert "reasoning" in result
                assert result["answer"] == "This is a test answer about Pikachu." 