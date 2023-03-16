import argparse

import gym
from gym import logger, spaces
from joblib import Parallel, delayed

from gym_adserver.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from gym_adserver.agents.random_agent import RandomAgent
from gym_adserver.agents.softmax_agent import SoftmaxAgent
from gym_adserver.agents.ucb1_agent import UCB1Agent
from gym_adserver.agents.thompson_sampling_agent import TSAgent
from gym_adserver.agents.etc_agent import ETCAgent

def setup_environment(env_name, num_ads, time_series_frequency):
    env = gym.make(env_name, num_ads=num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)
    return env

def render_environment(env, i, **kwargs):
    if kwargs['output_file'] is not None:
        kwargs['output_file'] = str(i) + "_" + kwargs['output_file']
    env.render(**kwargs)
    env.close()

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
    action_space = spaces.Discrete(args.num_ads)
    max_impressions = args.impressions
    seed = args.seed

    # Define your agents here
    agents = [
        RandomAgent(action_space=action_space),
        SoftmaxAgent(seed=seed, beta=5, max_impressions=max_impressions),
        UCB1Agent(action_space=action_space, seed=args.seed, c=2, max_impressions=max_impressions),
        EpsilonGreedyAgent(seed=seed, epsilon=0.1),
        TSAgent(action_space=action_space,seed=seed),
        ETCAgent(action_space=action_space,seed=seed,exploration_rounds=100)
    ]

    envs = []
    for agent in agents:
        logger.info('Starting {}'.format(agent.name))
        env = setup_environment(env_name=args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)

        # Simulation loop
        reward = 0
        observation = env.reset(options={"scenario_name": agent.name})
        envs.append(env)
        for i in range(args.impressions):
            # Action/Feedback
            ad_index = agent.act(observation=observation, reward=reward, done=False, ad_budgets = env.budgets)
            observation, reward, _, _ = env.step(ad_index)

    # Render result for each agent (NOTE: close all to quit)
    parallel = Parallel(n_jobs=-1)
    parallel(delayed(render_environment)(env, i, freeze=True, output_file=args.output_file) for i, env in enumerate(envs))