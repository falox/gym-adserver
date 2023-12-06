import argparse
import sys
import time

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import gym_adserver

class EpsilonGreedyAgent(object):
    def __init__(self, seed, epsilon):
        self.name = "epsilon-Greedy Agent"
        self.np_random = RandomState(seed)
        self.epsilon = epsilon

    def act(self, observation, reward, done, ad_budgets):
        ads, _, _ = observation
        
        # If all ad budgets are exhausted, return None
        if all(budget <= 0 for budget in ad_budgets):
            return None

        if np.random.uniform() < self.epsilon:
            # Exploration: choose randomly among the ads with available budget
            available_ads_indices = [i for i, budget in enumerate(ad_budgets) if budget > 0]
            ad_index = self.np_random.choice(available_ads_indices)
        else:
            # Exploitation: choose the ad with the highest CTR so far and available budget
            available_ctrs = [ad.ctr() if ad_budgets[i] > 0 else float('-inf') for i, ad in enumerate(ads)]
            ad_index = np.argmax(available_ctrs)

        return ad_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)

    # Setup the agent
    agent = EpsilonGreedyAgent(args.seed, args.epsilon)

    # Simulation loop
    reward = 0
    done = False
    observation = env.reset(seed=args.seed, options={"scenario_name": agent.name})
    for i in range(args.impressions):
        # Action/Feedback
        ad_index = agent.act(observation, reward, done, env.budgets)
        observation, reward, done, _ = env.step(ad_index)
        
        # Render the current state
        observedImpressions = observation[1]
        if observedImpressions % time_series_frequency == 0: 
            env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()