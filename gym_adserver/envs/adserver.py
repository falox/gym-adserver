from typing import Optional
import gym
from gym import logger, spaces
from gym.utils import seeding

import numpy as np
from numpy.random.mtrand import RandomState

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['toolbar'] = 'None'

from .ad import Ad
from .user import User

AD_TYPES = ['car', 'food', 'electronics', 'clothing']

class AdServerEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, num_ads, time_series_frequency, users=None, reward_policy=None, budgets=None, seed=None):        
        self.time_series_frequency = time_series_frequency        
        self.num_ads = num_ads
        self.reward_policy = reward_policy
        self.max_impressions = 300
        self.seed_value = seed
        self.seed(seed=self.seed_value)
        self.ads_random = np.random.default_rng(seed)
        
        self.users = [User(['car', 'electronics'], np_random=self.ads_random),
                      User(['food', 'clothing'], np_random=self.ads_random)]
        
        self.click_probabilities = [None] * num_ads
        self.bids = [None] * num_ads
        self.budgets = [None] * num_ads
        self.ads = [None] * num_ads
        self.revenues = [None] * num_ads
        self.types = [None] * num_ads
        

        # Initial state (can be reset later)
        self._generate_ads()
        clicks = 0
        impressions = 0
        self.state = (self.ads, impressions, clicks)
        self.ctr_time_series = []
        self.revenue_time_series = []

        # Environment OpenAI metadata
        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(num_ads) # index of the selected ad
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2, num_ads), dtype=np.float) # clicks and impressions, for each ad
            
    def _generate_ads(self):
        for i in range(self.num_ads):
            ad_type = self.ads_random.choice(AD_TYPES)
            click_probability = self.get_ad_type_click_probability(ad_type, 0)
            bid = self._generate_bid()
            budget = self._generate_budget()
            ad = Ad(id=i, click_probability=click_probability, bid=bid, budget=budget, type=ad_type)
            
            self.ads[i] = ad
            self.bids[i] = bid
            self.click_probabilities[i] = click_probability
            self.budgets[i] = budget
            self.revenues[i] = ad.revenue
            self.types[i] = ad.type

    def _generate_bid(self):
        # Your custom logic to generate bid based on ad_type
        return self.ads_random.uniform(0.01, 1)

    def _generate_budget(self):
        # Your custom logic to generate budget based on ad_type
        return self.ads_random.uniform(1000, 10000)

    def get_current_time_of_day(self):
        _, impressions, _ = self.state
        time_of_day = impressions / self.max_impressions
        return time_of_day

    def get_ad_type_click_probability(self, ad_type, time_of_day):
        # Define continuous functions for click probabilities depending on ad_type and time_of_day
        if ad_type == 'car':
            return 0.245 * np.sin(2 * np.pi * time_of_day - np.pi) + 0.255
        elif ad_type == 'food':
            return 0.245 * np.sin(4 * np.pi * time_of_day) + 0.255
        elif ad_type == 'electronics':
            return 0.245 * np.sin(6 * np.pi * time_of_day - 0.5 * - np.pi) + 0.255
        elif ad_type == 'clothing':
            return 0.245 * np.sin(8 * np.pi * time_of_day) + 0.255
    
    def step(self, action):
        ads, impressions, clicks = self.state
        
        if action is None:
            # All ad budgets are exhausted, terminate the episode
            return self.state, 0, True, {}
        
        # Deduct the cost of the click from the budget for the selected ad
        self.budgets[action] -= self.bids[action]
        
        # Update click probabilities
        # Select a random user
        user = self.ads_random.choice(self.users)
        for ad in ads:
            time_of_day = self.get_current_time_of_day()
            new_click_probability = self.get_ad_type_click_probability(ad.type, time_of_day) + user.bonus_click_probabilities[ad.type]
            ad.update_click_probability(new_click_probability)

        # Update clicks (if any)
        reward = self.draw_click(action)
        if reward == 1:
            clicks += 1
            ads[action].clicks += 1
            reward = self.bids[action]

        # Update impressions
        ads[action].impressions += 1
        impressions += 1

        # Update the ctr time series (for rendering)
        if impressions % self.time_series_frequency == 0:
            ctr = 0.0 if impressions == 0 else float(clicks / impressions)
            self.ctr_time_series.append(ctr)
        
        # Update the revenue time series (for rendering)
        if impressions % self.time_series_frequency == 0:
            total_revenue = sum(ad.revenue for ad in ads)
            self.revenue_time_series.append(total_revenue)

        self.state = (ads, impressions, clicks)

        return self.state, reward, False, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.seed(seed)
        self.scenario_name = options["scenario_name"]
        self._generate_ads()
        clicks = 0
        impressions = 0
        self.state = (self.ads, impressions, clicks)
        self.ctr_time_series = []
        self.revenue_time_series = []
        return self.state

    def render(self, mode='human', freeze=False, output_file=None): # pragma: no cover
        if mode != 'human':
            raise NotImplementedError
        ads, impressions, clicks = self.state
        ctr = 0.0 if impressions == 0 else float(clicks / impressions)

        logger.info('Scenario: {}, Impressions: {}, CTR: {}, Ads: {}'.format(self.scenario_name, impressions, ctr, ads))

        fig = plt.figure(num=self.scenario_name, figsize=(12, 6))
        grid_size = (5, 4)
        
        # Plot CTR time series
        plt.subplot2grid(grid_size, (0, 0), rowspan=1, colspan=4)
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

        # Plot impressions, bids, and budgets
        plt.subplot2grid(grid_size, (2, 0), rowspan=2, colspan=2)
        x = [ad.id for ad in ads]
        impressions = [ad.impressions for ad in ads]
        x_pos = [i for i, _ in enumerate(x)]
        plt.barh(x_pos, impressions)
        plt.ylabel("Ads")
        plt.xlabel("Impressions")
        plt.yticks(x_pos, x)
        for i, v in enumerate(impressions):
            plt.text(v, i, f"Bid: {self.bids[i]:.2f}, Budget: {self.budgets[i]:.2f}", va='center')

        # Plot CTRs and probabilities
        plt.subplot2grid(grid_size, (2, 2), rowspan=2, colspan=2)
        x = [ad.id for ad in ads]
        y = [ad.ctr() for ad in ads]
        y_2 = [ad.click_probability for ad in ads]
        x_pos = [i for i, _ in enumerate(x)]
        x_pos_2 = [i + 0.4 for i, _ in enumerate(x)]
        plt.ylabel("Ads")
        plt.xlabel("")
        plt.yticks(x_pos, x)
        plt.barh(x_pos, y, 0.4, label='Actual CTR')
        plt.barh(x_pos_2, y_2, 0.4, label='Probability')
        plt.legend(loc='upper right')

        # Plot Total Revenue
        plt.subplot2grid(grid_size, (1, 0), rowspan=1, colspan=4)
        x = [i for i, _ in enumerate(self.revenue_time_series)]
        y = self.revenue_time_series
        axes = plt.gca()
        axes.set_ylim([0,1500])
        plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
        plt.ylabel("Revenue")
        plt.xlabel("Impressions")
        plt.plot(x, y, marker='o')
        for x,y in zip(x,y):
            plt.annotate("{:.2f}".format(y), (x,y), textcoords="offset points", xytext=(0,10), ha='center')

        plt.tight_layout()

        if output_file is not None:
            fig.savefig(output_file)

        if freeze:
            if matplotlib.is_interactive():
                plt.ioff()
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(self.np_random.uniform(0.001, 0.1))  # Use the separate random generator for rendering                

    def draw_click(self, action):
        if self.reward_policy is not None:
            return self.reward_policy(action)

        if self.click_probabilities == [None] * self.num_ads:
            self.click_probabilities = [self.ads_random.uniform() * 0.5 for i in range(self.num_ads)]

        ad = self.ads[action]
        return self.ads_random.random() < ad.click_probability

    def close(self):
        plt.close()