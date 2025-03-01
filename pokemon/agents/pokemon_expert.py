from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from pokemon.core.config import ANTHROPIC_API_KEY


class BattleAnalysisResult(BaseModel):
    """Results of a Pokémon battle analysis"""
    winner: str = Field(description="The Pokémon predicted to win the battle")
    reasoning: str = Field(description="Detailed reasoning for the prediction")


class PokemonExpertAgent:
    """
    Agent that analyzes Pokémon battles and predicts winners based on stats and type matchups.
    Uses LangGraph for structured reasoning.
    """
    
    def __init__(self, 
        model: Optional[str] = "claude-3-5-haiku-20241022", 
        researcher_agent: Optional['ResearcherAgent'] = None # type: ignore
    ):
        """Initialize the Pokémon Expert Agent with tools and a model."""
        from pokemon.agents.researcher import ResearcherAgent

        self.llm = ChatAnthropic(
            model=model,
            api_key=ANTHROPIC_API_KEY
        )
        self.researcher = researcher_agent or ResearcherAgent()
        
        # Define tools the expert can use
        self.tools = [
            self.get_type_effectiveness,
            self.analyze_stats_comparison,
            self.compare_pokemon_data
        ]
        
        # System message that provides context about the agent's purpose
        system_message = """
        You are a Pokémon Battle Expert with extensive knowledge of Pokémon battles, type matchups, and competitive strategy.
        
        When analyzing battles between two Pokémon:
        1. Consider their types and type effectiveness (which types are strong/weak against others)
        2. Analyze their base stats (speed is often crucial in determining who attacks first)
        3. Consider their typical movesets and abilities
        4. Provide a clear prediction with detailed reasoning
        
        Type effectiveness chart (multipliers for attacking types against defending types):
        - Normal: weak to Fighting (x2); immune to Ghost (x0)
        - Fire: strong against Grass, Ice, Bug, Steel (x2); weak against Water, Rock, Fire, Dragon (x0.5)
        - Water: strong against Fire, Ground, Rock (x2); weak against Water, Grass, Dragon (x0.5)
        - Electric: strong against Water, Flying (x2); weak against Electric, Grass, Dragon (x0.5); no effect on Ground (x0)
        - Grass: strong against Water, Ground, Rock (x2); weak against Fire, Grass, Poison, Flying, Bug, Dragon, Steel (x0.5)
        - Ice: strong against Grass, Ground, Flying, Dragon (x2); weak against Fire, Water, Ice, Steel (x0.5)
        - Fighting: strong against Normal, Ice, Rock, Dark, Steel (x2); weak against Poison, Flying, Psychic, Bug, Fairy (x0.5); no effect on Ghost (x0)
        - Poison: strong against Grass, Fairy (x2); weak against Poison, Ground, Rock, Ghost (x0.5); no effect on Steel (x0)
        - Ground: strong against Fire, Electric, Poison, Rock, Steel (x2); weak against Grass, Bug (x0.5); no effect on Flying (x0)
        - Flying: strong against Grass, Fighting, Bug (x2); weak against Electric, Rock, Steel (x0.5)
        - Psychic: strong against Fighting, Poison (x2); weak against Psychic, Steel (x0.5); no effect on Dark (x0)
        - Bug: strong against Grass, Psychic, Dark (x2); weak against Fire, Fighting, Poison, Flying, Ghost, Steel, Fairy (x0.5)
        - Rock: strong against Fire, Ice, Flying, Bug (x2); weak against Fighting, Ground, Steel (x0.5)
        - Ghost: strong against Psychic, Ghost (x2); weak against Dark (x0.5); no effect on Normal (x0)
        - Dragon: strong against Dragon (x2); weak against Steel (x0.5); no effect on Fairy (x0)
        - Dark: strong against Psychic, Ghost (x2); weak against Fighting, Dark, Fairy (x0.5)
        - Steel: strong against Ice, Rock, Fairy (x2); weak against Fire, Water, Electric, Steel (x0.5)
        - Fairy: strong against Fighting, Dragon, Dark (x2); weak against Fire, Poison, Steel (x0.5)
        
        Always provide a clear winner and detailed reasoning.
        """
        
        # Create the agent executor
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=system_message
        )
    
    @tool
    def get_type_effectiveness(self, attacking_type: str, defending_types: List[str]) -> Dict[str, Any]:
        """
        Calculate type effectiveness multipliers for an attacking type against defending types.
        
        Args:
            attacking_type: The type of the attack (e.g., "Electric", "Fire")
            defending_types: List of the defending Pokémon's types (e.g., ["Grass", "Poison"])
            
        Returns:
            A dictionary with effectiveness information
        """
        # Type effectiveness chart (simplified version)
        effectiveness_chart = {
            "normal": {"fighting": 2.0, "ghost": 0.0},
            "fire": {"grass": 2.0, "ice": 2.0, "bug": 2.0, "steel": 2.0, 
                    "fire": 0.5, "water": 0.5, "rock": 0.5, "dragon": 0.5},
            "water": {"fire": 2.0, "ground": 2.0, "rock": 2.0,
                     "water": 0.5, "grass": 0.5, "dragon": 0.5},
            "electric": {"water": 2.0, "flying": 2.0, 
                        "electric": 0.5, "grass": 0.5, "dragon": 0.5, 
                        "ground": 0.0},
            "grass": {"water": 2.0, "ground": 2.0, "rock": 2.0,
                     "fire": 0.5, "grass": 0.5, "poison": 0.5, "flying": 0.5, 
                     "bug": 0.5, "dragon": 0.5, "steel": 0.5},
            "ice": {"grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0,
                   "fire": 0.5, "water": 0.5, "ice": 0.5, "steel": 0.5},
            "fighting": {"normal": 2.0, "ice": 2.0, "rock": 2.0, "dark": 2.0, "steel": 2.0,
                        "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "fairy": 0.5,
                        "ghost": 0.0},
            "poison": {"grass": 2.0, "fairy": 2.0,
                      "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5,
                      "steel": 0.0},
            "ground": {"fire": 2.0, "electric": 2.0, "poison": 2.0, "rock": 2.0, "steel": 2.0,
                      "grass": 0.5, "bug": 0.5,
                      "flying": 0.0},
            "flying": {"grass": 2.0, "fighting": 2.0, "bug": 2.0,
                      "electric": 0.5, "rock": 0.5, "steel": 0.5},
            "psychic": {"fighting": 2.0, "poison": 2.0,
                       "psychic": 0.5, "steel": 0.5,
                       "dark": 0.0},
            "bug": {"grass": 2.0, "psychic": 2.0, "dark": 2.0,
                   "fire": 0.5, "fighting": 0.5, "poison": 0.5, "flying": 0.5, 
                   "ghost": 0.5, "steel": 0.5, "fairy": 0.5},
            "rock": {"fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0,
                    "fighting": 0.5, "ground": 0.5, "steel": 0.5},
            "ghost": {"psychic": 2.0, "ghost": 2.0,
                     "dark": 0.5,
                     "normal": 0.0},
            "dragon": {"dragon": 2.0,
                      "steel": 0.5,
                      "fairy": 0.0},
            "dark": {"psychic": 2.0, "ghost": 2.0,
                    "fighting": 0.5, "dark": 0.5, "fairy": 0.5},
            "steel": {"ice": 2.0, "rock": 2.0, "fairy": 2.0,
                     "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
            "fairy": {"fighting": 2.0, "dragon": 2.0, "dark": 2.0,
                     "fire": 0.5, "poison": 0.5, "steel": 0.5}
        }
        
        # Normalize input
        attacking_type = attacking_type.lower()
        defending_types = [dt.lower() for dt in defending_types]
        
        # Calculate effectiveness
        multiplier = 1.0
        for defending_type in defending_types:
            if defending_type in effectiveness_chart.get(attacking_type, {}):
                multiplier *= effectiveness_chart[attacking_type][defending_type]
        
        effectiveness = "neutral"
        if multiplier > 1.0:
            effectiveness = "super effective"
        elif multiplier < 1.0 and multiplier > 0.0:
            effectiveness = "not very effective"
        elif multiplier == 0.0:
            effectiveness = "no effect"
        
        return {
            "attacking_type": attacking_type.capitalize(),
            "defending_types": [dt.capitalize() for dt in defending_types],
            "multiplier": multiplier,
            "effectiveness": effectiveness
        }

    @tool
    def analyze_stats_comparison(self, stats1: Dict[str, int], stats2: Dict[str, int]) -> Dict[str, Any]:
        """
        Compare the base stats of two Pokémon and determine advantages.
        
        Args:
            stats1: Dictionary of the first Pokémon's stats
            stats2: Dictionary of the second Pokémon's stats
            
        Returns:
            Analysis of the stat comparison
        """
        # Calculate total base stats
        total1 = sum(stats1.values())
        total2 = sum(stats2.values())
        
        # Determine stat advantages for each Pokémon
        advantages1 = []
        advantages2 = []
        
        for stat in stats1:
            if stat in stats2:
                if stats1[stat] > stats2[stat]:
                    advantages1.append(stat)
                elif stats2[stat] > stats1[stat]:
                    advantages2.append(stat)
        
        # Speed is often crucial in battles
        speed_advantage = None
        if "speed" in stats1 and "speed" in stats2:
            if stats1["speed"] > stats2["speed"]:
                speed_advantage = "pokemon1"
            elif stats2["speed"] > stats1["speed"]:
                speed_advantage = "pokemon2"
        
        return {
            "pokemon1_total": total1,
            "pokemon2_total": total2,
            "pokemon1_advantages": advantages1,
            "pokemon2_advantages": advantages2,
            "pokemon1_higher_total": total1 > total2,
            "pokemon2_higher_total": total2 > total1,
            "speed_advantage": speed_advantage
        }

    @tool
    def compare_pokemon_data(self, pokemon1_name: str, pokemon2_name: str) -> Dict[str, Any]:
        """
        Get and compare complete data for two Pokémon using the Researcher Agent.
        
        Args:
            pokemon1_name: Name of the first Pokémon
            pokemon2_name: Name of the second Pokémon
            
        Returns:
            Comprehensive comparison data
        """
        # Get data for both Pokémon
        pokemon1_data = self.researcher.get_pokemon_data(pokemon1_name)
        pokemon2_data = self.researcher.get_pokemon_data(pokemon2_name)
        
        # Check if either Pokémon wasn't found
        if isinstance(pokemon1_data, str) and "not found" in pokemon1_data:
            return {"error": f"Pokémon '{pokemon1_name}' not found"}
        if isinstance(pokemon2_data, str) and "not found" in pokemon2_data:
            return {"error": f"Pokémon '{pokemon2_name}' not found"}
        
        # Calculate type effectiveness in both directions
        type_effectiveness = {}
        if "types" in pokemon1_data and "types" in pokemon2_data:
            # Pokémon 1's attacks against Pokémon 2
            for attack_type in pokemon1_data["types"]:
                effect = self.get_type_effectiveness(attack_type, pokemon2_data["types"])
                type_effectiveness[f"{pokemon1_data['name']}_{attack_type}_vs_{pokemon2_data['name']}"] = effect
            
            # Pokémon 2's attacks against Pokémon 1
            for attack_type in pokemon2_data["types"]:
                effect = self.get_type_effectiveness(attack_type, pokemon1_data["types"])
                type_effectiveness[f"{pokemon2_data['name']}_{attack_type}_vs_{pokemon1_data['name']}"] = effect
        
        # Compare stats
        stats_comparison = {}
        if "stats" in pokemon1_data and "stats" in pokemon2_data:
            stats_comparison = self.analyze_stats_comparison(pokemon1_data["stats"], pokemon2_data["stats"])
        
        return {
            "pokemon1": pokemon1_data,
            "pokemon2": pokemon2_data,
            "type_effectiveness": type_effectiveness,
            "stats_comparison": stats_comparison
        }

    def determine_winner(self, pokemon1: str, pokemon2: str) -> Dict[str, str]:
        """
        Determine the winner between two Pokémon and provide reasoning.
        
        Args:
            pokemon1: Name of the first Pokémon
            pokemon2: Name of the second Pokémon
            
        Returns:
            Dictionary with winner and reasoning
        """
        # Format the query message
        query = f"Who would win in a battle between {pokemon1} and {pokemon2}? Analyze their types, stats, and abilities to determine a winner. Provide detailed reasoning."
        
        # Initialize the agent with the query
        response = self.agent_executor.invoke({"messages": [HumanMessage(content=query)]})
        
        # Extract the final answer
        final_answer = response["messages"][-1].content
        
        # Process the final answer to extract winner and reasoning
        # This is a simple approach - in a real system you might want more structured parsing
        # or prompt the LLM to return a more structured format
        if pokemon1.lower() in final_answer.lower() and "win" in final_answer.lower():
            winner = pokemon1
        elif pokemon2.lower() in final_answer.lower() and "win" in final_answer.lower():
            winner = pokemon2
        else:
            # If we can't clearly determine a winner from the text, default to the first one
            # but note that the determination is unclear
            winner = "can't determine who is the winner"
        
        return {
            "winner": winner,
            "reasoning": final_answer
        }
