#Shishir Tandale

import tensorflow as tf, numpy as np

#load glove pretrained embeddings
vocabSize = 1193514 #from `wc -l <filename>`
embeddingDim = 25 #we have options to later use 50d, 100d, and 200d vectors
glove = np.zeroes((vocabSize, embeddingDim))
glovefilename = "glove.twitter.27B.25d.txt"

lookup = {}
counter = 0
with open(glovefilename, "r") as gloveFile:
    embed = gloveFile.readline().split()
    lookup[embed[0]] = counter
    vect = [float(i) for i in embed[1:]]
    glove[counter] = vect
print("Loaded {} into memory".format(glovefilename))

W = tf.constant(embedding, name="W") #inefficient in memory

#load tweets and hashtags
def loadTweets():
    tweets = []
    hashtags = []
    with open("tweets.txt", "r") as tfile:
        tweets = tfile.readlines()
    with open("hashtags.txt", "r") as hfile:
        hashtags = hfile.readlines()
    return tweets, hashtags
#replace all words that don't appear in lookup with {UNK} or other token
def parseTweets(tweets):
    #remove punctuation/irregularities via regex
    #TODO filter for embedding
    return [tweet.lowercase().split() for tweet in tweets]
#create batches
def makeBatches(tweets, hashtags, batchsize):
    pass
def buildGraph():
    pass
#look up embeddings
#train hashtags (if needed?)
def forward(batches, hashtags):
    pass
#feed embeddings through covnet and feed forward
#backprop to predict hashtags
def nce_loss(loss):
    pass
