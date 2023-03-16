import argparse
import sys
import time

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import gym_adserver

class SoftmaxAgent(object):
    def __init__(self, seed, beta, max_impressions):
        self.name = "Softmax Agent"
        self.np_random = RandomState(seed)
        self.beta = beta
        self.max_impressions = max_impressions

    def act(self, observation, reward, done, ad_budgets):
        ads, current_impressions, _ = observation

        # Compute the temperature
        remaining_time = (self.max_impressions - current_impressions) / self.max_impressions
        temperature =  max(remaining_time ** self.beta, sys.float_info.min)
        
        # Softmax
        ctrs = [ad.ctr() for ad in ads]
        exponentials = np.exp((ctrs - np.max(ctrs)) / temperature) # the -max is for numerical stability, https://stackoverflow.com/a/40756996

        # Set the probability of choosing ads with exhausted budgets to zero
        probabilities = [exponential if ad_budgets[i] > 0 else 0 for i, exponential in enumerate(exponentials)]
        if sum(probabilities) != 0:
            probabilities /= sum(probabilities)
        else:
            return None

        # Weighted random selection
        action = self.np_random.choice(np.arange(len(ads)), p=probabilities)

        return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)

    # Setup the agent
    agent = SoftmaxAgent(args.seed, args.beta, args.impressions)

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