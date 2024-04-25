from citylearn.agents.base import BaselineAgent
from citylearn.agents.rbc import BasicRBC
from citylearn.agents.sac import SAC
from citylearn.agents.marlisa import MARLISA
from citylearn.citylearn import CityLearnEnv
import pandas as pd

# Function to train an agent
def train_agent(agent_name, episodes=1):
    # Initialize the environment
    env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=(agent_name != 'SAC' and agent_name != 'MARLISA'))
    
    # Select the agent based on agent_name
    if agent_name == 'Baseline':
        agent = BaselineAgent(env)
    elif agent_name == 'RBC':
        agent = BasicRBC(env)
    elif agent_name == 'SAC':
        agent = SAC(env)
    elif agent_name == 'MARLISA':
        agent = MARLISA(env)
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")
    
    # Train the agent
    agent.learn(episodes=episodes)
    
    # Evaluate the environment
    kpis = env.evaluate()
    kpis_df = pd.DataFrame(kpis)
    kpis_pivot = kpis_df.pivot(index='cost_function', columns='name', values='value').round(3)
    kpis_pivot = kpis_pivot.dropna(how='all')
    
    # Display the KPIs
    print(kpis_pivot)

# Example of how to use the function
agent_list=['Baseline','RBC','SAC','MARLISA']
for agent in agent_list:
    print('agent name',agent)
    train_agent('Baseline', episodes=1)
