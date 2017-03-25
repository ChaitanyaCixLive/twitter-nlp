#Shishir Tandale
import tensorflow as tf, numpy as np

class BaselineModel(object):
    def __init__(self, tweets, hashtags, embeddings):
        self.tweets = tweets
        self.hashtags = hashtags
        self.embeddings = embeddings
    #replace all words that don't appear in lookup with {UNK} or other token
    def parseTweets(tweets):
        #remove punctuation/irregularities via regex
        #TODO filter for embedding
        return [tweet.lowercase().split() for tweet in tweets]
    #create batches
    def makeBatches(tweets, hashtags, batchsize):
        pass
    #look up embeddings
    #train hashtags (if needed?)
    #feed embeddings through covnet and feed forward
    #backprop to predict hashtags
