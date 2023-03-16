import argparse
import sys
import time
import math

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import gym_adserver

class UCB1Agent(object):
    def __init__(self, action_space, seed, c, max_impressions):
        self.name = "UCB1 Agent"
        self.values = [0.00] * action_space.n
        self.np_random = RandomState(seed)
        # Exploration parameter
        self.c = c
        self.max_impressions = max_impressions
        self.prev_action = None
        self.rewards = []

    def act(self, observation, reward, done, ad_budgets):
        ads, impressions, _ = observation
        
        # If all ad budgets are exhausted, return None
        if all(budget <= 0 for budget in ad_budgets):
            return None

        # Update the value for the action of the previous act() call
        if self.prev_action != None:            
            n = ads[self.prev_action].impressions
            value = self.values[self.prev_action]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[self.prev_action] = new_value    
            self.rewards.append(reward)    
        
        # Test each ad once
        ad_index = 0
        for ad in ads:
            if ad.impressions == 0 and ad_budgets[ad_index] > 0:
                self.prev_action = ads.index(ad)
                return self.prev_action
            ad_index += 1

        # Compute the UCB values
        ucb_values = [0.0] * len(self.values)
        for i in range(len(ads)):
            if ad_budgets[i] > 0:
                exploitation = self.values[i]
                exploration = self.c * math.sqrt((math.log(impressions)) / float(ads[i].impressions))
                ucb_values[i] = exploitation + exploration
            else:
                ucb_values[i] = float('-inf')

        # Select the max UCB value
        self.prev_action = ucb_values.index(max(ucb_values))
        return self.prev_action

def compute_regret(agent, env, num_impressions):
    """
    Computes the regret for the given agent in the gym-adserver environment.

    :param agent: The Thompson Sampling agent
    :param env: The gym-adserver environment
    :param num_impressions: The number of impressions in the simulation
    :return: The regret for the agent
    """
    # Find the optimal ad index based on click_probabilities
    optimal_ad_index = np.argmax(env.click_probabilities)
    
    # Calculate the cumulative reward for the optimal action
    optimal_cumulative_reward = env.click_probabilities[optimal_ad_index] * num_impressions

    # Calculate the cumulative reward for the agent
    agent_cumulative_reward = sum(agent.rewards)

    # Compute regret for the agent
    regret = optimal_cumulative_reward - agent_cumulative_reward

    return regret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--c', type=float, default=2)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)

    # Setup the agent
    agent = UCB1Agent(env.action_space, args.seed, args.c, args.impressions)

    # Simulation loop
    reward = 0
    done = False
    observation = env.reset(seed=args.seed, options={"scenario_name": agent.name})
    for i in range(args.impressions):
        # Action/Feedback
        ad_index = agent.act(observation, reward, done, env.budgets)
        observation, reward, done, _ = env.step(ad_index)
        UCB_agent_regret = compute_regret(agent, env, i)
        print("Regret for UCB Sampling agent:", UCB_agent_regret)
        
        # Render the current state
        observedImpressions = observation[1]
        if observedImpressions % time_series_frequency == 0: 
            env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()