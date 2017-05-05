# Shishir Tandale
import json, re, sys, numpy as np
from models import baseline_model as bm, lstm_model as lm

tweet_embedding_map = {}
hashtag_embedding_map = {}
tweet_hashtag_map = {}
hashtag_tweet_map = {}

tweet_text_map = {}
hashtag_text_map = {}
tweet_id_map = {}
hashtag_id_map = {}

class Hashtag(object):
    current_idi = 0
    def __init__(self, hashtag_text):
        if hashtag_text not in hashtag_text_map:
            self.text = hashtag_text
            self.embedding = None
            self.id = Hashtag.current_id()
            hashtag_text_map[hashtag_text] = self
            hashtag_id_map[self.id] = self
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
        self.numTweets  = numTweets
        jsontxt_sized = jsontxt.readlines()[:numTweets]
        print("Parsing text into JSON Object")
        self.tweetJSONObjs = [json.loads(line) for line in jsontxt_sized]
    def progress_init(self, message):
        self.progress_n = 0.
        self.progress_step = 100./self.numTweets
        self.progress_message = message
    def progress(self, data=None):
        self.progress_n += self.progress_step
        if self.progress_n < 100:
            sys.stdout.write('\r{}: {}%'.format(self.progress_message, round(self.progress_n,3)))
        else:
            sys.stdout.write('\r'+' '*(len(self.progress_message)+10)+'\r')
        sys.stdout.flush()
        return data
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

def loadGlove(embeddingFile, gloveSize, gloveDim):
    lookup = {}
    counter = 0
    glove = np.zeros((gloveSize, gloveDim))
    with open(embeddingFile, "r") as ef:
        for line in ef:
            embed = line.split(' ')
            word, *embeddingVector = embed
            lookup[word] = counter
            vect = [float(i) for i in embeddingVector]
            glove[counter] = vect
            counter += 1
    return glove, lookup

def main():
    testFile = "../data/twitter-nlp/json/cache-0.json"
    gloveFile = "../data/twitter-nlp/embeddings/glove.twitter.27B.25d.txt"
    vocabSize = 100000 # `wc -l <testFile>`
    gloveSize = 1193514
    embeddingDim = 25 # must match up with glove
    numHashtags = 50 # num most common hashtags to embed
    numHashtags_print = 5 # used in test and debug methods

    json = open(testFile)
    # json = tw_download.getTweets("shishirtandale", 1000)
    tweets, hashtags = TwitterJSONParse(json, vocabSize).parseJSON()
    print("Num tweets: {}, Num unique hashtags: {}".format(len(tweet_hashtag_map.keys()), len(hashtag_tweet_map.keys())))

    print("Sorting and processing hashtags")
    # map each hashtag to the number of tweets its associated with, sort, then reverse the list
    sortedHashtags = (sorted([(len(hashtag_tweet_map[ht_id]), hashtag_id_map[ht_id]) for ht_id in hashtag_id_map.keys()]))[-1::-1]
    _, justHashtagsSorted = zip(*sortedHashtags)
    print("{} most common hashtags: {}".format(numHashtags_print, sortedHashtags[:numHashtags_print]))

    print("Loading Glove Embeddings")
    glove25, glove_lookup = loadGlove(gloveFile, gloveSize, embeddingDim)

    print("Initializing BaselineModel")
    blm = bm.BaselineModel(tweets, justHashtagsSorted, glove25, gloveSize, glove_lookup, vocabSize, embeddingDim, numHashtags, tweet_embedding_map, hashtag_embedding_map, tweet_hashtag_map, hashtag_tweet_map, tweet_text_map, hashtag_text_map, tweet_id_map, hashtag_id_map)

    print("Initialization finished, collecting results")
#    tsne_wv, tsne_vocab = blm.finishedHTEmbeddings, justHashtagsSorted[:numHashtags]
#    print("Sample Embedding: {} => {}".format(tsne_vocab[0], tsne_wv[0]))

if __name__ == "__main__":
    main()
