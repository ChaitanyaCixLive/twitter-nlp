# Shishir Tandale
import json, re, sys
from utils.reverse_dict import Symmetric_Dictionary as sdict

class TwitterProperty(object):
    idi = 0
    _store = sdict() #supports reversibility
    _set = set()
    def __init__(self):
        #ensures self points to the latest version of this object
        self = TwitterProperty.register(self)
    @classmethod
    def register(cls, ref):
        #see if ref has been encountered before, if so, return that reference
        if hash(ref) not in cls._store:
            cls.idi += 1
            index = cls.idi
            cls._store[hash(ref)] = (ref, index)
            ref.index = index
        else:
            ref, _ = cls._store[hash(ref)]
        if ref not in cls.set(): #make sure our data structs are updated
            cls.set().add(ref)
        return ref
    @classmethod
    def set(cls):
        return cls._set
class User(TwitterProperty):
    _set = set() #required for User's own _set
    def __init__(self, username, tweets=None):
        self.username = username
        self.tweets = tweets
        self.embedding = None
        self = User.register(self)
    def __str__(self):
        return self.username
    def __hash__(self):
        return hash(str(self))
class Hashtag(TwitterProperty):
    _set = set() #required for Hashtag's own _set
    def __init__(self, hashtag, tweets=[]):
        self.text = hashtag
        self.popularity = 0
        self.embedding = None
        self.parent_tweets = []
        self = Hashtag.register(self)
        #add to popularity after retrieving latest copy
        self.popularity += 1
    def __str__(self):
        return self.text
    def __hash__(self):
        return hash(str(self))
    #use hashtag popularity as metric for comparison
    def __lt__(self, ot):
        return self.popularity < ot.popularity
class Tweet(TwitterProperty):
    _set = set() #required for Tweet's own _set
    def __init__(self, text, hashtags=None):
        self.text = text
        self.hashtags = hashtags
        self.embedding = None
        self = Tweet.register(self)
    def __str__(self):
        return self.text #modify to include hashtags
    def __hash__(self):
        return hash(str(self))
    def has_hashtags(self):
        return len(self.hashtags) > 0
    def has_text(self):
        return str(self) is not None and len(str(self)) > 0
class TwitterJSONParse(object):
    def __init__(self, jsontxt, num_tweets, show_progress=True):
        self.num_tweets  = num_tweets
        self.show_progress = show_progress
        jsontxt_resized = jsontxt.readlines()[:num_tweets]
        if self.show_progress:
            from utils.progress import Progress
            with Progress("Parsing text into JSON Object", count=num_tweets, precision=2) as (u,_,_):
                self.tweet_JSON_objs = [u(json.loads(line)) for line in jsontxt_resized]
        else:
            print("Parsing text into JSON Object")
            self.tweet_JSON_objs = [json.loads(line) for line in jsontxt_resized]
    def process_tweets(self):
        # for more documentation, visit:
        # https://dev.twitter.com/overview/api/tweets
        def extract_text(tweet_json):
            r_hashtag = "([#].*)"
            r_twlink = "(http://t[.]co.*)"
            # remove all hashtags and twitter links from text
            tweet_text = re.sub(r_hashtag, "<hashtag>", tweet_json["text"].lower())
            tweet_text = re.sub(r_twlink, "<url>", tweet_text)
            hashtags = [(Hashtag(h["text"].lower())) for h in tweet_json["entities"]["hashtags"]]
            tweet_obj = Tweet(tweet_text, hashtags)
            for tag in hashtags: tag.parent_tweets.append(tweet_obj)
            return tweet_obj
        #TODO add progress
        for obj in self.tweet_JSON_objs:
            extract_text(obj)
        print("Tweets extracted.")
