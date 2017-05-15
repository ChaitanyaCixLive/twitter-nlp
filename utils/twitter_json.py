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

class Twitter_JSON_Parse(object):
    def __init__(self, json_txt,
                show_progress=False, replace_hashtags=True,
                replace_user_refs=False, replace_links=True):
        import json
        self.show_progress = show_progress
        self.replace_links = replace_links
        self.replace_hashtags = replace_hashtags
        self.replace_user_refs = replace_user_refs
        #parse text into json objects
        if self.show_progress:
            from utils.progress import Progress
            with Progress("Parsing text into JSON Object", len(json_txt)) as up:
                self.tweet_JSON_objs = [u(json.loads(line)) for line in json_txt]
        else:
            print("Parsing text into JSON Object.")
            self.tweet_JSON_objs = [json.loads(line) for line in json_txt]
        #extract text from tweets
        self.process_tweets()

    def process_tweets(self):
        import re
        # for more documentation, visit:
        # https://dev.twitter.com/overview/api/tweets
        def apply_regex(text):
            # remove hashtags, links, user_refs from tweets
            if self.replace_hashtags:
                #accept hashtags within words, optional trailing :
                r_hashtag = r"#\w+\b"
                text = re.sub(r_hashtag, "<hashtag>", text)
            if self.replace_links:
                #replace both twitter and pbs.twimg links not within words (malformed)
                #{4,} avoids selecting non twitter links with short base urls
                r_twlink = r"\bhttps?:\/\/(?:(?:twitter)|(?:pbs\.twimg)|(?:t))[./?=\w]{4,}\b"
                text = re.sub(r_twlink, "<url>", text)
            if self.replace_user_refs:
                #user refs can be 1 to 15 \w characters, not starting within words (avoid emails)
                r_usrrefs = r"\B@\w{1,15}:?\b"
                text = re.sub(r_usrrefs, "<user>", text)
            return text

        def extract_text(tweet_json):
            orig_text = tweet_json["text"].lower().strip()
            tweet_text = apply_regex(orig_text)
            tweet_obj = Tweet(tweet_text, orig_text)
            #get, process, and package hashtags (this also stores and counts them)
            hashtags = [Hashtag(h["text"].lower()) for h in tweet_json["entities"]["hashtags"]]
            #connect hashtags to tweet and vice versa
            tweet_obj.register_hashtags(hashtags)
            [h.register_tweet(tweet_obj) for h in hashtags]

        if self.show_progress:
            with Progress("Extracting tweets", len(self.tweet_JSON_objs)) as up:
                [up(extract_text(tweet_obj)) for tweet_obj in self.tweet_JSON_objs]
        else:
            print("Extracting tweets.")
            [extract_text(tweet_obj) for tweet_obj in self.tweet_JSON_objs]
