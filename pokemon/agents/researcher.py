import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from core.config import GEMINI_API_KEY

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


class PokemonInfo(BaseModel):
    """Information about a Pokemon"""
    name: str = Field(description="Name of the Pokemon")
    id: int = Field(description="Pokemon ID number")
    types: List[str] = Field(description="Types of the Pokemon")
    stats: Dict[str, int] = Field(description="Base stats of the Pokemon")
    height: int = Field(description="Height of the Pokemon in decimeters")
    weight: int = Field(description="Weight of the Pokemon in hectograms")
    abilities: List[str] = Field(description="Abilities of the Pokemon")


class ResearcherAgent:
    BASE_URL = "https://pokeapi.co/api/v2/pokemon/"
    
    def __init__(self, model: Optional[str] = "gemini-2.0-flash"):
        """Initialize the researcher agent with tools for interacting with the PokéAPI."""
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=GEMINI_API_KEY
        )
        self.tools = [
            self.get_pokemon_data,
            self.compare_pokemon
        ]
        
        # Create a system message that provides context about the agent's purpose
        system_message = """
        You are a Pokemon Research Assistant specialized in retrieving accurate information from the PokéAPI.
        When asked about Pokemon, use the provided tools to fetch relevant details.
        Always provide factual information based on the API results.
        Format your responses clearly for both humans and other agents.
        """
        
        # Create the agent executor
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_message
        )
    
    @tool
    def get_pokemon_data(self, pokemon_name: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a specific Pokemon by name or ID.
        
        Args:
            pokemon_name: The name or ID of the Pokemon to look up (case-insensitive).
            
        Returns:
            A dictionary containing the Pokemon's data.
        """
        try:
            # Normalize input to lowercase for consistent API calls
            pokemon_name = pokemon_name.lower().strip()
            response = requests.get(f"{self.BASE_URL}{pokemon_name}")
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and format the relevant information
            pokemon_info = {
                "name": data["name"].capitalize(),
                "id": data["id"],
                "types": [t["type"]["name"].capitalize() for t in data["types"]],
                "stats": {
                    stat["stat"]["name"].replace("-", "_"): stat["base_stat"] 
                    for stat in data["stats"]
                },
                "height": data["height"],
                "weight": data["weight"],
                "abilities": [ability["ability"]["name"].replace("-", " ").capitalize() 
                              for ability in data["abilities"]]
            }
            
            return pokemon_info
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return f"Pokemon '{pokemon_name}' not found. Please check the spelling."
            return f"Error fetching Pokemon data: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


    @tool
    def compare_pokemon(self, pokemon_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple Pokemon by retrieving their data and organizing it for comparison.
        
        Args:
            pokemon_names: List of Pokemon names to compare.
            
        Returns:
            A dictionary containing comparable data for all specified Pokemon.
        """
        results = {}
        
        for name in pokemon_names:
            pokemon_data = self.get_pokemon_data(name)
            if isinstance(pokemon_data, dict):
                results[pokemon_data["name"]] = pokemon_data
            else:
                results[name] = {"error": pokemon_data}
                
        return results

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query about Pokemon by leveraging the agent to retrieve and format information.
        
        Args:
            question: The question about Pokemon to be answered.
            
        Returns:
            A dictionary containing the answer and any supporting information.
        """
        # Create a human message with the question
        messages = [HumanMessage(content=question)]
        
        # Invoke the agent
        result = self.agent_executor.invoke({"messages": messages})
        
        # Extract the last AI message for the response
        final_answer = result["messages"][-1].content
        
        return {
            "answer": final_answer,
        }


