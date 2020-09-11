import argparse
import sys
import time

import gym
from gym import wrappers, logger

import gym_adserver

class RandomAgent(object):

    def __init__(self, action_space):
        self.name = "Random Agent"
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)

    # Setup the agent
    agent = RandomAgent(env.action_space)

    # Simulation loop
    reward = 0
    done = False
    observation = env.reset(agent.name)
    for i in range(args.impressions):
        # Action/Feedback
        action = agent.act(observation, reward, done)
        observation, reward, done, _ = env.step(action)
        
        # Render the current state
        observedImpressions = observation[1]
        if observedImpressions % time_series_frequency == 0: 
            env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()