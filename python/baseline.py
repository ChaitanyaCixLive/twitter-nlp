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
    pass
#replace all words that don't appear in lookup with {UNK} or other token
def parseTweets(tweets):
    pass
#create batches
def makeBatches(tweets, hashtags):
    pass
#look up embeddings
#train hashtags (if needed?)
#feed embeddings through covnet and feed forward
#backprop to predict hashtags
