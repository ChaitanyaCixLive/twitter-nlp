import json, re, sys, numpy as np
#import baseline_model

class TwitterJSONParse(object):
    def __init__(self, json):
        self.json = json
    def containsHashtags(self, tweet):
        return len(tweet['entities']['hashtags']) > 0
    def getHashtags(self, tweet):
        # returns empty string if no hashtags
        return " ".join(["#"+h["text"] for h in tweet['entities']['hashtags']])
    def parseJSON(self):
		# for more documentation, visit:
		# https://dev.twitter.com/overview/api/tweets
		# only saves useful tweets and hashtags for embeddings
        def format(tweet):
            r_hashtag = "([#].*)"
            r_twlink = "(http[:][/]{2}t[.]co.*)"
            # remove all hashtags and twitter links from text
            text = (re.sub("|".join([r_hashtag,r_twlink]), "", tweet["text"]))
            hashtags = (self.getHashtags(tweet))
            return text, hashtags
        def isValid(tweet, hashtags):
            # text is only hashtags, or a link. not useful
            if hashtags == "" or text == "":
                return False
            else:
                return True
        self.tweetObjs = [json.loads(line) for line in self.json]
        self.formattedTweets = [format(tw) for tw in self.tweetObjs]
        self.filteredTweets = [tw for tw in self.formattedTweets if isValid]
        #package up for return
        tweets, hashtags = zip(*self.filteredTweets)
        return tweets, hashtags
def loadGlove(embeddingFile, vocabSize, embeddingDim):
    lookup = {}
    counter = 0
    glove = np.zeros((vocabSize, embeddingDim))
    with open(embeddingFile, "r") as ef:
        embed = ef.readline().split()
        lookup[embed[0]] = counter
        vect = [float(i) for i in embed[1:]]
        glove[counter] = vect
    return embeddings, lookup
def main(*stdin):
    testFile = "data/json/cache-0-first300.json"
    json = open(testFile)
    tweets, hashtags = TwitterJSONParse(json).parseJSON()
    print("Num Tweets: {}".format(len(tweets)))
    glove25, lookup = loadGlove("data/embeddings/glove.twitter.27B.25d.txt", 1193514, 25)
    blm = BaselineModel(tweets, hashtags, glove25)

if __name__ == "__main__":
    main()
