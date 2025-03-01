from typing import Dict, List, Any, Optional, TypedDict, Literal
import re

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from pokemon.core.config import ANTHROPIC_API_KEY


class AgentState(TypedDict):
    """Represents the state of the agent workflow."""
    messages: List[Any]  # Human and AI messages
    question: str  # Original user question
    next_step: Optional[str]  # Next step to take (if any)
    pokemon_data: Optional[Dict[str, Any]]  # Data from researcher
    pokemon_names: Optional[List[str]]
    pokemon_name: Optional[str]
    battle_result: Optional[Dict[str, str]]  # Battle result from expert
    final_answer: Optional[Dict[str, Any]]  # Final answer to return to user


class SupervisorAgent:
    """
    Orchestrates the workflow between different agents based on query type.
    
    Decides whether to:
    1. Answer directly (for general knowledge)
    2. Delegate to the Researcher (for Pokemon data)
    3. Delegate to the Pokemon Expert (for battle analysis)
    """
    
    def __init__(
        self, 
        researcher_agent: Optional['ResearcherAgent'] = None, # type: ignore
        expert_agent: Optional['PokemonExpertAgent'] = None, # type: ignore
        model: Optional[str] = "claude-3-5-haiku-20241022"
    ):
        from pokemon.agents.researcher import ResearcherAgent
        from pokemon.agents.pokemon_expert import PokemonExpertAgent


        """Initialize the Supervisor Agent with specialized agents and tools."""
        self.researcher = researcher_agent or ResearcherAgent(model=model)
        self.expert = expert_agent or PokemonExpertAgent(model=model)
        self.llm = ChatAnthropic(
            model=model,
            api_key=ANTHROPIC_API_KEY
        )
        
        # Define tools the supervisor can use
        self.tools = []
        
        # System message for the supervisor agent
        system_message = """
        You are a Pokemon Knowledge Supervisor that determines how to best answer user questions.
        
        For any question, analyze whether:
        1. It's a GENERAL knowledge question that you can answer directly
        2. It requires POKEMON DATA to be fetched from an external API
        3. It's asking about a POKEMON BATTLE between two Pokemon
        
        Be efficient and only use specialized agents when necessary.
        """
        
        # Create the agent executor for classification
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=system_message,
            name="Supervisor Agent"
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
            
    def _classify_question(self, state: AgentState) -> AgentState:
        """
        Classify the type of question and determine the next step using structured output.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the next_step field set
        """
        from pydantic import BaseModel, Field
        
        # Define a Pydantic model for the structured output
        class QuestionClassification(BaseModel):
            """Classification of a Pokemon-related question."""
            question_type: Literal["general_knowledge", "pokemon_research", "pokemon_data", "battle_analysis"] = Field(
                ..., 
                description="The type of question being asked")
            pokemon_names: List[str] = Field(
                default_factory=list, 
                description="Names of Pokemon mentioned in battle questions (2+ for battles)")
            pokemon_name: str = Field(
                default="", 
                description="Name of the Pokemon for data questions (single Pokemon)")
            confidence: float = Field(
                default=1.0, 
                description="Confidence in the classification (0.0 to 1.0)")
        
        # Get the user's question
        question = state["question"]
        
        # Create a prompt for classification
        prompt = f"""Classify this Pokemon-related question into one of these categories:
        1. "general_knowledge" - General questions that don't need special Pokemon data
        2. "pokemon_research" - Questions about Pokemon that need research but not specific data lookup
        3. "pokemon_data" - Questions about specific Pokemon's stats, abilities, etc. (extract the Pokemon name)
        4. "battle_analysis" - Questions about which Pokemon would win in a battle (extract both Pokemon names)

        Question: {question}"""
        
        try:
            # Use structured output to classify the question
            structured_llm = self.llm.with_structured_output(QuestionClassification)
            result = structured_llm.invoke(prompt)
            
            # Set the next step based on the classification
            state["next_step"] = result.question_type
            
            # Store relevant Pokemon names
            if result.question_type == "battle_analysis" and len(result.pokemon_names) >= 2:
                state["pokemon_names"] = [name.lower() for name in result.pokemon_names]
            elif result.question_type == "pokemon_data" and result.pokemon_name:
                state["pokemon_name"] = result.pokemon_name.lower()
            
            # If confidence is low, default to pokemon_research as fallback
            if result.confidence < 0.7:
                state["next_step"] = "pokemon_research"
                
        except Exception as e:
            # Fallback if structured output fails - use pokemon_research as the safest option
            state["next_step"] = "pokemon_research"
        
        return state
    
    def _direct_answer(self, state: AgentState) -> AgentState:
        """
        Answer a general knowledge question directly.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the answer
        """
        # Get the user's question
        question = state["question"]
        
        # Call the LLM to answer the question
        response = self.llm.invoke([HumanMessage(content=question)])
        
        # Extract the answer
        answer = response.content
        
        # Store the answer in the state
        state["final_answer"] = {"answer": answer}
        
        return state
    
    def _pokemon_research(self, state: AgentState) -> AgentState:
        """
        Research Pokemon information using the researcher agent.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the research results
        """
        # Get the user's question
        question = state["question"]
        
        # Call the researcher agent to get the answer
        result = self.researcher.query(question)
        
        # Store the result in the state
        state["pokemon_data"] = result
        state["final_answer"] = result
        
        return state
    
    def _pokemon_data(self, state: AgentState) -> AgentState:
        """
        Get specific Pokemon data using the researcher agent.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the Pokemon data
        """
        # Get the Pokemon name from the state
        pokemon_name = state.get("pokemon_name", "")
        
        # If we don't have a Pokemon name, use the question
        if not pokemon_name:
            state = self._pokemon_research(state)
            return state
        
        # Get the Pokemon data
        from pokemon.agents.researcher import get_pokemon_data
        pokemon_data = get_pokemon_data.invoke(pokemon_name)
        
        # Check if we got a valid response
        if isinstance(pokemon_data, dict):
            # Store the data in the state
            state["pokemon_data"] = pokemon_data
            state["final_answer"] = pokemon_data
        else:
            # If we couldn't get the data, use the researcher to answer
            state = self._pokemon_research(state)
        
        return state
    
    def _battle_analysis(self, state: AgentState) -> AgentState:
        """
        Analyze a battle between two Pokemon using the expert agent.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the battle analysis
        """
        # Get the Pokemon names from the state
        pokemon_names = state.get("pokemon_names", [])
                
        # Get the battle analysis
        battle_result = self.expert.determine_winner(pokemon_names[0], pokemon_names[1])
        
        # Store the result in the state
        state["battle_result"] = battle_result
        state["final_answer"] = battle_result
        
        return state
    
    def _decide_next_step(self, state: AgentState) -> Literal["direct_answer", "pokemon_research", "pokemon_data", "battle_analysis", "END"]:
        """
        Decide the next step based on the state.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            The name of the next step
        """
        # If we've already determined the next step, return it
        next_step = state.get("next_step")
        
        # If we have a final answer, we're done
        if state.get("final_answer"):
            return "END"
        
        # Return the next step
        return next_step
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the workflow graph for the supervisor agent.
        
        Returns:
            A StateGraph representing the workflow
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add the nodes for each step
        workflow.add_node("classify_question", self._classify_question)
        workflow.add_node("direct_answer", self._direct_answer)
        workflow.add_node("pokemon_research", self._pokemon_research)
        workflow.add_node("get_pokemon_data", self._pokemon_data)
        workflow.add_node("battle_analysis", self._battle_analysis)
        
        # Set the entry point
        workflow.set_entry_point("classify_question")
        
        # Add the edges to connect the nodes
        workflow.add_conditional_edges(
            "classify_question",
            self._decide_next_step,
            {
                "direct_answer": "direct_answer",
                "pokemon_research": "pokemon_research",
                "pokemon_data": "get_pokemon_data",
                "battle_analysis": "battle_analysis"
            }
        )
        
        # Connect all output nodes to END
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("pokemon_research", END)
        workflow.add_edge("get_pokemon_data", END)
        workflow.add_edge("battle_analysis", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question from the user and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            A dictionary containing the answer and reasoning
        """
        # Initialize the state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "next_step": None,
            "pokemon_data": None,
            "pokemon_names": None,
            "pokemon_name": None,
            "battle_result": None,
            "final_answer": None
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Get the final answer
        result = final_state["final_answer"]
        
        # Format the response consistently
        if isinstance(result, dict):
            # For battle analysis
            if "winner" in result and "reasoning" in result:
                return {
                    "answer": f"{result['winner']} would win the battle.",
                    "reasoning": result["reasoning"]
                }
            # For direct answers that already have the right format
            elif "answer" in result:
                return result
            # For pokemon data responses
            else:
                answer = result.get("answer", str(result))
                reasoning = None
                return {"answer": answer, "reasoning": reasoning}
        else:
            # Handle case where result is a string or other type
            return {"answer": str(result), "reasoning": None}