# Shishir Tandale
import numpy as np
from utils.progress import Progress
from utils.twitter_json import Hashtag, Tweet, TwitterJSONParse
from utils.twitter_download import TweepyClient
from models import baseline_model as bm, lstm_model as lm

def loadGlove(embeddingFile, gloveSize, gloveDim):
    lookup = {}
    counter = 0
    glove = np.zeros((gloveSize, gloveDim))
    with open(embeddingFile, "r") as ef:
        with Progress("Loading Glove Embeddings", count=gloveSize) as (update,_,_):
            for line_num, line in enumerate(ef):
                word, *embeddingVector = line.split(' ')
                lookup[word] = line_num
                glove[line_num] = [float(i) for i in embeddingVector]
                update()
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

    tjp = TwitterJSONParse(json, vocabSize)
    # load all dicts
    tweet_embedding_map, hashtag_embedding_map, tweet_hashtag_map, hashtag_tweet_map, \
    tweet_text_map, hashtag_text_map, tweet_id_map, hashtag_id_map = tjp.dicts

    tweets, hashtags = tjp.parseJSON()
    print("Num tweets: {}, Num unique hashtags: {}".format(len(tweet_hashtag_map.keys()), len(hashtag_tweet_map.keys())))

    print("Sorting and processing hashtags")
    # map each hashtag to the number of tweets its associated with, sort, then reverse the list
    sortedHashtags = (sorted([(len(hashtag_tweet_map[ht_id]), hashtag_id_map[ht_id]) for ht_id in hashtag_id_map.keys()]))[-1::-1]
    _, justHashtagsSorted = zip(*sortedHashtags)
    print("{} most common hashtags: {}".format(numHashtags_print, sortedHashtags[:numHashtags_print]))

    glove25, glove_lookup = loadGlove(gloveFile, gloveSize, embeddingDim)

    print("Initializing BaselineModel")
    #TODO CURRENT model breaking bug -- dicts need to be reorganized
    #blm = bm.BaselineModel(tweets, justHashtagsSorted, glove25, gloveSize, glove_lookup, vocabSize, embeddingDim, numHashtags, tjp)

    print("Initialization finished, collecting results")
#    tsne_wv, tsne_vocab = blm.finishedHTEmbeddings, justHashtagsSorted[:numHashtags]
#    print("Sample Embedding: {} => {}".format(tsne_vocab[0], tsne_wv[0]))

if __name__ == "__main__":
    main()
