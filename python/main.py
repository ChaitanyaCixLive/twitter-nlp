# Shishir Tandale
import json, re, sys, numpy as np
import baseline_model as bm
import tw_download

class Hashtag(object):
    hashtag_map = None
    def __init__(self, hashtag, tweet):
        self.tweet = tweet
        self.hashtag = hashtag
        self.embedding = None
        # check if hashtag is in map, if not, adds it
        # uses a list attached to the dict to store tweets, unordered
        if Hashtag.hashtag_map == None:
            Hashtag.hashtag_map = {self:[self.tweet]}
        else:
            if self not in Hashtag.hashtag_map.keys():
                Hashtag.hashtag_map[self] = [self.tweet]
            else:
                Hashtag.hashtag_map[self].append(self.tweet)
        # ensures we only keep the first, primary instance
        self = Hashtag.hashtag_map[self]
    def __repr__(self):
        return self.hashtag
    def __hash__(self):
        return hash(self.__repr__())
    def __eq__(self, other):
        return self.hashtag == other.hashtag
class Tweet(object):
    #TODO add dict pointing back to associated Hashtags
    #TODO add way to map all tweets to their embeddings
    def __init__(self, text):
        self.text = text
        self.embedding = None
        self.hashtags = None
    def __repr__(self):
        return self.text

class TwitterJSONParse(object):
    def __init__(self, jsontxt, numTweets):
        self.numTweets  = numTweets
        jsontxt_sized = jsontxt.readlines()[:numTweets]
        self.progress_init("Parsing text into JSON Object")
        self.tweetJSONObjs = [self.progress(json.loads(line)) for line in jsontxt_sized]
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

            hashtags = [Hashtag(h["text"].lower(), tweetObj) for h in tweet["entities"]["hashtags"]]
            return tweetObj, hashtags
        # used for progress indicator
        self.progress_init("Formatting and extracting hashtags")
        formattedTweets = [self.progress(extractText(obj)) for obj in self.tweetJSONObjs]
        filteredTweets = [(tweet, hashtags) for (tweet, hashtags) in formattedTweets if (hashtags != [] and tweet.text != "")]
        # package up for retur
        tweets, hashtags = zip(*filteredTweets)
        return tweets, hashtags, Hashtag.hashtag_map

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
    vocabSize = 10000 # `wc -l <testFile>`
    gloveSize = 1193514
    embeddingDim = 25 # must match up with glove
    numHashtags = 50 # num most common hashtags to embed
    numHashtags_print = 5 # used in test and debug methods

    json = open(testFile)
    # json = tw_download.getTweets("shishirtandale", 1000)
    tweets, hashtags, hashtag_map = TwitterJSONParse(json, vocabSize).parseJSON()
    print("Num tweets: {}, Num unique hashtags: {}".format(len(tweets), len(hashtag_map.keys())))

    print("Sorting and processing hashtags")
    # map each hashtag to the number of tweets its associated with, sort, then reverse the list
    sortedHashtags = sorted([(len(Hashtag.hashtag_map[key]), key.hashtag, key) for key in Hashtag.hashtag_map.keys()])[-1::-1]
    justHashtagsSorted = [hashtag for _, _, hashtag in sortedHashtags]
    print("{} most common hashtags: {}".format(numHashtags_print, sortedHashtags[:numHashtags_print]))

    print("Loading Glove Embeddings")
    glove25, glove_lookup = loadGlove(gloveFile, gloveSize, embeddingDim)

    print("Initializing BaselineModel")
    blm = bm.BaselineModel(tweets, justHashtagsSorted, hashtag_map, glove25, gloveSize, glove_lookup, vocabSize, embeddingDim, numHashtags)

    print("Initialization finished, collecting results")
    tsne_wv, tsne_vocab = blm.finishedHTEmbeddings, justHashtagsSorted[:numHashtags]
    print("Sample Embedding: {} => {}".format(tsne_vocab[0], tsne_wv[0]))

if __name__ == "__main__":
    main()
