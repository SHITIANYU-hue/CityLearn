from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import BaselineAgent
from citylearn.agents.rbc import BasicRBC
from citylearn.agents.sac import SAC
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.graph_marlisa import graphMARLISA

import pandas as pd
from citylearn.reward_function import RewardFunction
import numpy as np
# Define CustomReward
import numpy as np

class CustomReward(RewardFunction):
    def __init__(self, env):
        super().__init__(env)

    def calculate(self):
        reward_list = []
        for b in self.env.buildings:
            # Assuming these are available as methods or properties on the building object
            temperature = b.current_temperature
            humidity = b.current_humidity
            air_quality = b.current_air_quality  # Could be PM2.5, CO2, VOC, etc.
            oxygen_level = b.current_oxygen_level
            noise_level = b.current_noise_level

            # Define target levels for each component
            temp_target = 22  # Ideal temperature in degrees Celsius
            humidity_target = 50  # Ideal humidity in percentage
            air_quality_target = 0.1  # Ideal level of PM2.5 or equivalent metric
            oxygen_target = 21  # Ideal oxygen level in percentage
            noise_target = 40  # Ideal noise level in dB

            # Calculate penalties for deviation from targets
            temp_penalty = -abs(temperature - temp_target)
            humidity_penalty = -abs(humidity - humidity_target)
            air_quality_penalty = -abs(air_quality - air_quality_target)
            oxygen_penalty = -abs(oxygen_level - oxygen_target)
            noise_penalty = -abs(noise_level - noise_target)

            # Weighing the penalties (optional weights could be adjusted based on specific priorities)
            temp_weight = 1.0
            humidity_weight = 0.8
            air_quality_weight = 1.2
            oxygen_weight = 0.5
            noise_weight = 0.5

            # Calculate total reward for each building
            total_reward = (temp_penalty * temp_weight +
                            humidity_penalty * humidity_weight +
                            air_quality_penalty * air_quality_weight +
                            oxygen_penalty * oxygen_weight +
                            noise_penalty * noise_weight)

            reward_list.append(total_reward)

        # Sum rewards from all buildings
        total_reward = sum(reward_list)
        return [total_reward]

# Function to train an agent with an option to use custom reward
def train_agent(agent_name, episodes=1, use_custom_reward=False):
    # Initialize the environment
    env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=(agent_name != 'SAC' and agent_name != 'MARLISA'))
    
    # Conditionally set the custom reward function
    if use_custom_reward:
        env.reward_function = CustomReward(env)
    
    # Select the agent based on agent_name
    if agent_name == 'Baseline':
        agent = BaselineAgent(env)
    elif agent_name == 'RBC':
        agent = BasicRBC(env)
    elif agent_name == 'SAC':
        agent = SAC(env)
    elif agent_name == 'MARLISA':
        agent = MARLISA(env)

    elif agent_name == 'graphMARLISA':
        agent = graphMARLISA(env)
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
agent_list=['Baseline','RBC','SAC','MARLISA','graphMARLISA']
for agent in agent_list:
    print('agent name',agent)
    train_agent(agent, episodes=1)
