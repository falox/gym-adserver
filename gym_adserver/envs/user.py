import numpy as np

class User:
    def __init__(self, preferences, np_random=None):
        self.preferences = preferences
        self.np_random = np_random if np_random is not None else np.random.default_rng()
        self.bonus_click_probabilities = self.get_bonus_click_probabilities()

    def get_bonus_click_probabilities(self):
        bonus_probabilities = {}
        for ad_type in ['car', 'food', 'electronics', 'clothing']:
            bonus_probabilities[ad_type] = self.np_random.uniform(0, 0.1) if ad_type in self.preferences else 0
        return bonus_probabilities

