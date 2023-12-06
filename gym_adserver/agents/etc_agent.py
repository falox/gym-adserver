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

class ETCAgent(object):
    def __init__(self, action_space, seed, exploration_rounds):
        self.name = "ETC Agent"
        self.values = [0.0] * action_space.n
        self.np_random = np.random.RandomState(seed)
        self.exploration_rounds = exploration_rounds
        self.prev_action = None
        self.rewards = []

    def act(self, observation, reward, done, ad_budgets):
        ads, impressions, _ = observation
        
        # If all ad budgets are exhausted, return None
        if all(budget <= 0 for budget in ad_budgets):
            return None

        # Update the value for the action of the previous act() call
        if self.prev_action is not None:
            n = ads[self.prev_action].impressions
            value = self.values[self.prev_action]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[self.prev_action] = new_value
            self.rewards.append(reward)

        # During the exploration phase, select each ad with available budget a fixed number of times
        for ad in ads:
            ad_index = ads.index(ad)
            if ad.impressions < self.exploration_rounds and ad_budgets[ad_index] > 0:
                self.prev_action = ad_index
                return self.prev_action

        # After the exploration phase, commit to the ad with the highest estimated value
        # with a non-exhausted budget
        available_values = [value if ad_budgets[i] > 0 else float('-inf') for i, value in enumerate(self.values)]
        self.prev_action = available_values.index(max(available_values))
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
    parser.add_argument('--exploration_rounds', type=int, default=20)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)

    # Setup the agent
    agent = ETCAgent(env.action_space, args.seed, args.exploration_rounds)

    # Simulation loop
    reward = 0
    done = False
    observation = env.reset(seed=args.seed, options={"scenario_name": agent.name})
    for i in range(args.impressions):
        # Action/Feedback
        ad_index = agent.act(observation, reward, done, env.budgets)
        observation, reward, done, _ = env.step(ad_index)
        ETC_agent_regret = compute_regret(agent, env, i)
        print("Regret for ETC agent:", ETC_agent_regret)
        
        # Render the current state
        observedImpressions = observation[1]
        if observedImpressions % time_series_frequency == 0: 
            env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()