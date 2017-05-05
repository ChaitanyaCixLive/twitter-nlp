# Shishir Tandale
import json, re, sys
from utils.progress import Progress

class Hashtag(object):
    current_idi = 0

    @property
    def hashtag_text_map(self):
        if self._hashtag_text_map is None:
            self._hashtag_text_map = {}
        return self._hashtag_text_map
    @property
    def hashtag_id_map(self):
        if self._hashtag_id_map is None:
            self._hashtag_id_map = {}
        return self._hashtag_id_map

    def __init__(self, hashtag_text):
        if hashtag_text not in self.hashtag_text_map:
            self.text = hashtag_text
            self.embedding = None
            self.id = Hashtag.current_id()
            self.hashtag_text_map[hashtag_text] = self
            self.hashtag_id_map[self.id] = self
        else:
            self = hashtag_text_map[hashtag_text]
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        return self.text == other.text
    def __repr__(self):
        return self.text
    def __lt__(self, other):
        return self.__repr__() < other.__repr__()
    @staticmethod
    def getHashtag(text):
        if text not in hashtag_text_map:
            hashtagObj = Hashtag(text)
            hashtag_text_map[text] = hashtagObj
            return hashtagObj
        else:
            return hashtag_text_map[text]
    @staticmethod
    def assocHashtag(text, tweetObj):
        hashtag = Hashtag.getHashtag(text)
        hashtag.add_link(tweetObj)
        tweetObj.add_link(hashtag)
        return hashtag
    @staticmethod
    def current_id():
        Hashtag.current_idi += 1
        return Hashtag.current_idi-1
    def add_link(self, tweet_obj):
        if self.id in hashtag_tweet_map:
            hashtag_tweet_map[self.id].append(tweet_obj.id)
        else:
            hashtag_tweet_map[self.id] = [tweet_obj.id]
class Tweet(object):
    current_idi = 0
    @property
    def tweet_id_map(self):
        if self._tweet_id_map is None:
            self._tweet_id_map = {}
        return self._tweet_id_map
    def __init__(self, text):
        self.text = text
        self.embedding = None
        self.id = Tweet.current_id()
        tweet_id_map[self.id] = self
    @staticmethod
    def getTweet(text):
        if text not in tweet_text_map:
            tweetObj = Tweet(text)
            tweet_text_map[text] = tweetObj
            return tweetObj
        else:
            return tweet_text_map[text]
    @staticmethod
    def current_id():
        Tweet.current_idi += 1
        return Tweet.current_idi - 1
    def add_link(self, hashtag_obj):
        if self.id in tweet_hashtag_map:
            tweet_hashtag_map[self.id].append(hashtag_obj.id)
        else:
            tweet_hashtag_map[self.id] = [hashtag_obj.id]

class TwitterJSONParse(object):
    def __init__(self, jsontxt, numTweets):
        self.tweet_embedding_map = {}
        self.hashtag_embedding_map = {}
        self.tweet_hashtag_map = {}
        self.hashtag_tweet_map = {}
        self.tweet_text_map = {}
        self.hashtag_text_map = {}
        self.tweet_id_map = {}
        self.hashtag_id_map = {}
        self.dicts = (self.tweet_embedding_map, self.hashtag_embedding_map, self.tweet_hashtag_map, self.hashtag_tweet_map, \
            self.tweet_text_map, self.hashtag_text_map, self.tweet_id_map, self.hashtag_id_map)

        self.numTweets  = numTweets
        jsontxt_sized = jsontxt.readlines()[:numTweets]
        with Progress("Parsing text into JSON Object", count=numTweets) as (update,_,_):
            self.tweetJSONObjs = [update(json.loads(line)) for line in jsontxt_sized]
    def parseJSON(self):
        # for more documentation, visit:
        # https://dev.twitter.com/overview/api/tweets
        # only saves useful tweets and hashtags for embeddings
        def extractText(tweet):
            r_hashtag = "([#].*)"
            r_twlink = "(http://t[.]co.*)"
            # remove all hashtags and twitter links from text
            text = re.sub(r_hashtag, "<hashtag>", tweet["text"].lower())
            text = re.sub(r_twlink, "<url>", text)
            tweetObj = Tweet(text)
            hashtags = [Hashtag.assocHashtag(h["text"].lower(), tweetObj) for h in tweet["entities"]["hashtags"]]
            return tweetObj, hashtags
        # used for progress indicator
        print("Formatting and extracting hashtags")
        formattedTweets = [extractText(obj) for obj in self.tweetJSONObjs]
        filteredTweets = [(tweet, hashtags) for (tweet, hashtags) in formattedTweets if (hashtags != [] and tweet.text != "")]
        # package up for return
        tweets, hashtags = zip(*filteredTweets)
        return tweets, hashtags
