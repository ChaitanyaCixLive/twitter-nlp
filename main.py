# Shishir Tandale
import numpy as np
from utils.progress import Progress
from utils.twitter_json import Tweet, Hashtag, User, TwitterJSONParse
from utils.twitter_download import TweepyClient
from models import baseline_model as bm, lstm_model as lm

def loadGlove(embeddingFile, gloveSize, gloveDim):
    lookup = {}
    counter = 0
    glove = np.zeros((gloveSize, gloveDim))
    with open(embeddingFile, "r") as ef:
        with Progress("Loading Glove Embeddings", count=gloveSize) as (u,_,_):
            for line_num, line in enumerate(ef):
                word, *embeddingVector = line.split(' ')
                lookup[word] = line_num
                glove[line_num] = [float(i) for i in embeddingVector]
                u()
    return glove, lookup

def main():
    test_file = "../data/twitter-nlp/json/cache-0.json"
    glove_file = "../data/twitter-nlp/embeddings/glove.twitter.27B.25d.txt"
    vocab_size = 100000 # `wc -l <testFile>`
    glove_size = 1193514
    embedding_dim = 25 # must match up with glove
    num_hashtags = 50 # num most common hashtags to embed
    num_hashtags_print = 5 # used in test and debug methods

    json = open(test_file)
    twitter_parse = TwitterJSONParse(json, vocab_size, show_progress=True)
    twitter_parse.process_tweets()
    tweets, hashtags = list(Tweet.set()), list(Hashtag.set())
    print(f"Num tweets: {len(tweets)}, Num unique hashtags: {len(hashtags)}")

    sorted_hts = sorted((hashtags))[-1::-1]
    #for ht in sorted_hts[:num_hashtags_print]:
        #print(ht, ht.popularity)
    glove25, glove_lookup = loadGlove(glove_file, glove_size, embedding_dim)
    print("Initializing BaselineModel")
    glove_params = (glove25, glove_size, glove_lookup)
    training_params = (vocab_size, embedding_dim, num_hashtags)
    blm = bm.BaselineModel(tweets, sorted_hts, glove_params, training_params)

if __name__ == "__main__":
    main()
