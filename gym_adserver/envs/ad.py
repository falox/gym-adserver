class Ad:
    def __init__(self, click_probability, id, bid, budget, type, impressions=0, clicks=0):
        self.id = str(id)
        self.impressions = impressions
        self.clicks = clicks
        self.bid = bid
        self.budget = budget
        self.type = type
        self.click_probability = click_probability
        
    def update_click_probability(self, click_probability):
        self.click_probability = click_probability

    def ctr(self):
        """Gets the CTR (Click-through rate) for this ad.

        Returns:
            float: Returns the CTR (between 0 and 1)
        """
        return 0.0 if self.impressions == 0 else float(self.clicks / self.impressions)
    
    @property
    def revenue(self):
        return self.clicks * self.bid

    def __repr__(self):
        return "({0}/{1})".format(self.clicks, self.impressions)
    
    def __str__(self):
        return "Ad: {0}, CTR: {1:.4f}".format(self.id, self.ctr())

    def __eq__(self, other) : 
        return self.id == other.id and self.impressions == other.impressions and self.clicks == other.clicks
