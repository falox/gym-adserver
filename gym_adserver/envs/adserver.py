import gym
from gym import logger, spaces
from gym.utils import seeding

import numpy as np
from numpy.random.mtrand import RandomState

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['toolbar'] = 'None'

from .ad import Ad

class AdServerEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, num_ads, time_series_frequency, reward_policy=None):        
        self.time_series_frequency = time_series_frequency        
        self.num_ads = num_ads
        self.reward_policy = reward_policy
        self.click_probabilities = None

        # Initial state (can be reset later)
        ads = [Ad(i) for i in range(num_ads)]
        clicks = 0
        impressions = 0
        self.state = (ads, impressions, clicks)
        self.ctr_time_series = []

        # Environment OpenAI metadata
        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(num_ads) # index of the selected ad
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2, num_ads), dtype=np.float) # clicks and impressions, for each ad

    def seed(self, seed=None): # pragma: no cover
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ads, impressions, clicks = self.state

        # Update clicks (if any)
        reward = self.draw_click(action)
        if reward == 1:
            clicks += 1
            ads[action].clicks += 1

        # Update impressions
        ads[action].impressions += 1
        impressions += 1

        # Update the ctr time series (for rendering)
        if impressions % self.time_series_frequency == 0:
            ctr = 0.0 if impressions == 0 else float(clicks / impressions)
            self.ctr_time_series.append(ctr)

        self.state = (ads, impressions, clicks)

        return self.state, reward, False, {}

    def reset(self, scenario_name):
        self.scenario_name = scenario_name

        ads = [Ad(i) for i in range(self.num_ads)]
        clicks = 0
        impressions = 0                
        self.state = (ads, impressions, clicks)
        self.ctr_time_series = []
        return self.state

    def render(self, mode='human', freeze=False, output_file=None): # pragma: no cover
        if mode != 'human':
            raise NotImplementedError

        ads, impressions, clicks = self.state
        ctr = 0.0 if impressions == 0 else float(clicks / impressions)

        logger.info('Impressions: {}, CTR: {}, Ads: {}'.format(impressions, ctr, ads))

        fig = plt.figure(num=self.scenario_name,figsize=(9, 6))
        grid_size = (5, 2)
        
        # Plot CTR time series 
        plt.subplot2grid(grid_size, (0, 0), rowspan=2, colspan=2)
        x = [i for i, _ in enumerate(self.ctr_time_series)]
        y = self.ctr_time_series        
        axes = plt.gca()
        axes.set_ylim([0,None])
        plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
        plt.ylabel("CTR")
        plt.xlabel("Impressions")
        plt.plot(x, y, marker='o')
        for x,y in zip(x,y):
            plt.annotate("{:.2f}".format(y), (x,y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot impressions
        plt.subplot2grid(grid_size, (2, 0), rowspan=3, colspan=1)
        x = [ad.id for ad in ads]
        impressions = [ad.impressions for ad in ads]
        x_pos = [i for i, _ in enumerate(x)]
        plt.barh(x_pos, impressions)
        plt.ylabel("Ads")
        plt.xlabel("Impressions")
        plt.yticks(x_pos, x)

        # Plot CTRs and probabilities
        plt.subplot2grid(grid_size, (2, 1), rowspan=3, colspan=1)
        x = [ad.id for ad in ads]
        y = [ad.ctr() for ad in ads]
        y_2 = self.click_probabilities
        x_pos = [i for i, _ in enumerate(x)]
        x_pos_2 = [i + 0.4 for i, _ in enumerate(x)]
        plt.ylabel("Ads")
        plt.xlabel("")
        plt.yticks(x_pos, x)        
        plt.barh(x_pos, y, 0.4, label='Actual CTR')
        plt.barh(x_pos_2, y_2, 0.4, label='Probability')
        plt.legend(loc='upper right')
                
        plt.tight_layout()

        if output_file is not None:
            fig.savefig(output_file)

        if freeze:
            # Keep the plot window open
            # https://stackoverflow.com/questions/13975756/keep-a-figure-on-hold-after-running-a-script
            if matplotlib.is_interactive(): 
                plt.ioff()
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.001)                 

    def draw_click(self, action):
        if self.reward_policy is not None:
            return self.reward_policy(action)

        if self.click_probabilities is None:
            self.click_probabilities = [self.np_random.uniform() * 0.5 for i in range(self.num_ads)]

        return 1 if self.np_random.uniform() <= self.click_probabilities[action] else 0

    def close(self):
        plt.close()