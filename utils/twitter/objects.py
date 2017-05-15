# Shishir Tandale

class TwitterProperty(object):
    #this super class helps maintain static data structures allowing for efficient
    #data ingestion. all subclasses need to do is call super().__init__() after
    #enough data to calculate __str__ (or an overridden __hash__) is stored.
    def __init__(self):
        try:
            #check if attributes are initialized and initialize them if needed
            self.store
            self.set
            self.class_id
        except AttributeError:
            self.store = dict()
            self.set = set()
            self.class_id = 0
        #update index and membership in data structs
        if self not in self.store:
            self.index = self.class_id
            self.store[self] = (self, self.index)
            self.class_id = self.index + 1
        else:
            nobj, _ = self.store[self]
            self.__dict__ = nobj.__dict__.copy()
        if self not in self.set:
            self.set.add(self)
    #implementing eq and hash allows str(self) to be used for set membership testing
    def __eq__(self, other):
        return hash(self) == hash(other)
    def __hash__(self):
        return hash(str(self))
    #property getters and setters for controlling static attributes
    @property
    def set(self): return self.__class__.set
    @set.setter
    def set(self, value): self.__class__.set = value
    @property
    def store(self): return self.__class__.store
    @store.setter
    def store(self, value): self.__class__.store = value
    @property
    def class_id(self): return self.__class__.idi
    @class_id.setter
    def class_id(self, value): self.__class__.idi = value

class User(TwitterProperty):
    def __init__(self, username, tweets=None):
        self.username = username
        self.tweets = tweets
        self.embedding = None
        super().__init__()
    def __str__(self):
        return self.username

class Hashtag(TwitterProperty):
    def __init__(self, hashtag):
        self.text = hashtag
        self.popularity = 0
        self.embedding = None
        self.parent_tweets = []
        super().__init__()
        #add to popularity after retrieving latest copy
        self.popularity += 1

    def __str__(self):
        return self.text
    def __lt__(self, ot):
        #use hashtag popularity as metric for comparison for sorted()
        return self.popularity < ot.popularity
    def register_tweet(self, tweet):
        self.parent_tweets.append(tweet)

class Tweet(TwitterProperty):
    def __init__(self, text, orig_text):
        self.text = text
        self.orig_text = orig_text
        self.hashtags = None
        self.embedding = None
        super().__init__()
    def __str__(self):
        return self.orig_text
    def has_hashtags(self):
        return len(self.hashtags) > 0
    def has_text(self):
        return str(self) is not None and len(str(self)) > 0
    def register_hashtags(self, hashtags):
        self.hashtags = hashtags
