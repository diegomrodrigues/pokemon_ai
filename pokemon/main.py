from pokemon.core import config
from pokemon.agents.researcher import ResearcherAgent
from pokemon.agents.pokemon_expert import PokemonExpertAgent
from pokemon.agents.supervisor import SupervisorAgent

# Create the specialized agents
researcher = ResearcherAgent()
expert = PokemonExpertAgent(researcher_agent=researcher)

# Create the supervisor agent
supervisor = SupervisorAgent(researcher_agent=researcher, expert_agent=expert)
