# Shishir Tandale
import json, re, sys, numpy as np
# import baseline_model

class Hashtag(object):
    hashtag_map = None
    def __init__(self, hashtag, tweet_text):
        self.tweet = tweet_text
        self.hashtag = hashtag
        # check if hashtag is in map, if not, adds it
        # uses a list attached to the dict to store tweets, unordered
        if Hashtag.hashtag_map == None:
            Hashtag.hashtag_map = {self.hashtag:[self.tweet]}
        else:
            if self.hashtag not in Hashtag.hashtag_map.keys():
                Hashtag.hashtag_map[self.hashtag] = [self.tweet]
            else:
                Hashtag.hashtag_map[self.hashtag].append(self.tweet)
        # ensures we only keep the first, primary instance
        self = Hashtag.hashtag_map[self.hashtag]
class TwitterJSONParse(object):
    def __init__(self, jsontxt, numTweets):
        self.numTweets  = numTweets
        self.progress_init("Parsing text into JSON Object")
        self.tweetObjs = [self.progress(json.loads(line)) for line in jsontxt]
    def progress_init(self, message):
        self.progress_n = 0.
        self.progress_message = message
    def progress(self, data=None):
        self.progress_n += 100./self.numTweets
        if self.progress_n < 100:
            sys.stdout.write('\r{0}: {1}%'.format(self.progress_message, round(self.progress_n,2)))
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

            hashtags = [Hashtag(h["text"].lower(), text) for h in tweet["entities"]["hashtags"]]
            return text, hashtags
        # used for progress indicator
        self.progress_init("Formatting and extracting hashtags")
        formattedTweets = [self.progress(extractText(obj)) for obj in self.tweetObjs]
        self.progress_init("Filtering tweets")
        filteredTweets = [(tweet, hashtags) for (tweet, hashtags) in formattedTweets if self.progress(hashtags != [] and tweet != "")]
        # package up for retur
        tweets, hashtags = zip(*filteredTweets)
        return tweets, hashtags, Hashtag.hashtag_map

def loadGlove(embeddingFile, vocabSize, embeddingDim):
    lookup = {}
    counter = 0
    glove = np.zeros((vocabSize, embeddingDim))
    with open(embeddingFile, "r") as ef:
        embed = ef.readline().split()
        lookup[embed[0]] = counter
        vect = [float(i) for i in embed[1:]]
        glove[counter] = vect
    return glove, lookup
def write_stdout(str):
    sys.stdout.write(str)
    sys.stdout.flush()

def main():
    testFile = "data/json/cache-0-first100000.json"
    # testFile = "data/json/cache-0.json"
    gloveFile = "data/embeddings/glove.twitter.27B.25d.txt"
    vocabSize = 100000 # `wc -l <testFile>`
    embeddingDim = 25 # must match up with glove
    numHashtags = 500 # num most common hashtags to embed
    numHashtags_print = 100 # used in test and debug methods

    json = open(testFile)
    tweets, hashtags, hashtag_map = TwitterJSONParse(json, vocabSize).parseJSON()
    print("Num tweets: {}, Num unique hashtags: {}".format(len(tweets), len(hashtag_map.keys())))
    # map each hashtag to the number of tweets its associated with, sort, then reverse the list
    sortedHashtags = sorted([(len(hashtag_map[key]), key) for key in hashtag_map.keys()])[-1::-1]
    justHashtagsSorted = [hashtag for _, hashtag in sortedHashtags]
    print("{} most common hashtags: {}".format(numHashtags_print, sortedHashtags[:numHashtags_print]))

    glove25, glove_lookup = loadGlove(gloveFile, vocabSize, embeddingDim)
    # blm = BaselineModel(tweets, justHashtagsSorted, hashtag_map, glove25, glove_lookup, vocabSize, embeddingDim, numHashtags)

if __name__ == "__main__":
    main()
