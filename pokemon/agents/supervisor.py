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
            state_modifier=system_message
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _check_pokemon_battle_impl(self, question: str) -> bool:
        """Determine if the question is about a Pokemon battle."""
        # Check for battle-related keywords
        battle_keywords = ["battle", "fight", "versus", "vs", "win", "defeat"]
        
        # Check if the question mentions two Pokemon
        has_battle_keywords = any(keyword in question.lower() for keyword in battle_keywords)
        
        return has_battle_keywords
    
    def _check_pokemon_question_impl(self, question: str) -> bool:
        """Determine if the question is related to Pokemon information."""
        # Check for question-related keywords
        pokemon_keywords = ["pokemon", "pokedex", "ability", "type", "species"]
        
        # Check if the question is about Pokemon generally
        has_pokemon_keywords = any(keyword in question.lower() for keyword in pokemon_keywords)
        
        return has_pokemon_keywords
    
    def _check_pokemon_data_impl(self, question: str) -> bool:
        """Determine if the question is asking for specific Pokemon data."""
        # Check for data-related keywords
        data_keywords = ["stats", "height", "weight", "type", "ability", "information", "details"]
        
        # Check if the question is requesting specific Pokemon data
        has_data_keywords = any(keyword in question.lower() for keyword in data_keywords)
        
        return has_data_keywords
    
    def _extract_pokemon_names(self, question: str) -> List[str]:
        """
        Extract Pokemon names from a battle question.
        
        Args:
            question: The user's question
            
        Returns:
            List of Pokemon names found in the question
        """
        # Common patterns to extract Pokemon names from battle questions
        patterns = [
            r'(?:between|of)\s+(\w+)\s+(?:and|vs\.?|versus)\s+(\w+)',  # between X and Y
            r'(\w+)\s+(?:vs\.?|versus)\s+(\w+)',  # X vs Y
            r'(?:would|will|could)\s+(\w+)\s+(?:beat|defeat|win against)\s+(\w+)',  # Would X beat Y
            r'(?:would|will|could)\s+(\w+)\s+(?:or)\s+(\w+)\s+(?:win)',  # Would X or Y win
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return [match.group(1).lower(), match.group(2).lower()]
        
        # If no pattern matches, look for capitalized words that might be Pokemon names
        # This is a simple heuristic and might not be accurate
        capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', question)
        if len(capitalized_words) >= 2:
            return [capitalized_words[0].lower(), capitalized_words[1].lower()]
            
        return []
    
    def _extract_pokemon_name(self, question: str) -> str:
        """
        Extract a single Pokemon name from a data question.
        
        Args:
            question: The user's question
            
        Returns:
            The extracted Pokemon name
        """
        # Common patterns to extract a Pokemon name
        patterns = [
            r'(?:about|of)\s+(\w+)(?:\s|\'|$)',  # about Pikachu, of Charizard
            r'(?:is|are)\s+(\w+)(?:\'s|\s)',  # is Pikachu's, are Charizard
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # If no pattern matches, look for capitalized words that might be Pokemon names
        capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', question)
        if capitalized_words:
            return capitalized_words[0].lower()
            
        return ""
    
    def _classify_question(self, state: AgentState) -> AgentState:
        """
        Classify the type of question and determine the next step.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the next_step field set
        """
        # Get the user's question
        question = state["question"]
        
        # Call the agent executor to classify the question
        response = self.agent_executor.invoke({
            "messages": [HumanMessage(content=f"Classify this question: {question}")]
        })
        
        # Extract the agent's answer
        agent_response = response["messages"][-1].content
        
        # Determine the next step based on the classification
        if self._check_pokemon_battle_impl(question):
            pokemon_names = self._extract_pokemon_names(question)
            if len(pokemon_names) >= 2:
                # Store the Pokemon names in the state
                state["pokemon_names"] = pokemon_names
                state["next_step"] = "battle_analysis"
            else:
                # If we couldn't extract Pokemon names, default to research
                state["next_step"] = "pokemon_research"
        elif self._check_pokemon_data_impl(question):
            pokemon_name = self._extract_pokemon_name(question)
            if pokemon_name:
                # Store the Pokemon name in the state
                state["pokemon_name"] = pokemon_name
                state["next_step"] = "pokemon_data"
            else:
                # If we couldn't extract a Pokemon name, default to research
                state["next_step"] = "pokemon_research"
        elif self._check_pokemon_question_impl(question):
            state["next_step"] = "pokemon_research"
        else:
            # This is a general knowledge question
            state["next_step"] = "direct_answer"
        
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
        pokemon_data = self.researcher.get_pokemon_data(pokemon_name)
        
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
        
        # If we don't have two Pokemon names, use the question
        if len(pokemon_names) < 2:
            state = self._pokemon_research(state)
            return state
        
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
            A dictionary containing the answer
        """
        # Initialize the state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "next_step": None,
            "pokemon_data": None,
            "battle_result": None,
            "final_answer": None
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Return the final answer
        return final_state["final_answer"]