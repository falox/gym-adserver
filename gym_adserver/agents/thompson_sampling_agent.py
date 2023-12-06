import argparse
import sys
import time
import math

import numpy as np
from numpy.random.mtrand import RandomState
from scipy.stats import beta

import gym
from gym import wrappers, logger

import gym_adserver

stationary=True

class TSAgent(object):
    def __init__(self, action_space, seed):
        self.name = "TS Agent"
        self.successes = [0] * action_space.n
        self.failures = [0] * action_space.n
        self.np_random = np.random.RandomState(seed)
        self.prev_action = None
        self.rewards = []

    def act(self, observation, reward, done, ad_budgets):
        ads, _, _ = observation
        
        # If all ad budgets are exhausted, return None
        if all(budget <= 0 for budget in ad_budgets):
            return None

        # Update the successes and failures for the action of the previous act() call
        if self.prev_action is not None:
            self.successes[self.prev_action] += reward
            self.failures[self.prev_action] += 1 - reward

        # Select the ad with the highest sampled beta value and available budget
        sampled_values = [0.0] * len(ads)
        for i, ad in enumerate(ads):
            if ad_budgets[i] > 0:
                a = self.successes[i] + 1
                b = self.failures[i] + 1
                sampled_values[i] = self.np_random.beta(a, b)
            else:
                sampled_values[i] = float('-inf')

        self.prev_action = sampled_values.index(max(sampled_values))
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
    agent = TSAgent(env.action_space, args.seed)

    # Simulation loop
    reward = 0
    done = False
    observation = env.reset(seed=args.seed, options={"scenario_name": agent.name})
    for i in range(args.impressions):
        # Action/Feedback
        ad_index = agent.act(observation, reward, done, env.budgets)
        observation, reward, done, _ = env.step(ad_index)
        thompson_agent_regret = compute_regret(agent, env, i)
        print("Regret for Thompson Sampling agent:", thompson_agent_regret)
        
        # Render the current state
        observedImpressions = observation[1]
        if observedImpressions % time_series_frequency == 0: 
            env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()