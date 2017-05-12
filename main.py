# Shishir Tandale
import numpy as np
import tensorflow as tf

from collections import defaultdict

from utils.progress import Progress
from utils.twitter_json import Tweet, Hashtag, User, TwitterJSONParse
from utils.twitter_download import TweepyClient
from models.baseline_model import Baseline_Model as Baseline
from models.hybrid_vae_model import Hybrid_VAE_Model as VAE

def loadGlove(embeddingFile, gloveSize, gloveDim):
    import tensorflow as tf
    lookup = defaultdict(lambda: -1, {})
    counter = 0
    glove = np.zeros((gloveSize, gloveDim))
    with open(embeddingFile, "r") as ef: #Progress("Loading Glove Embeddings", count=gloveSize) as (u,_,_):
            for line_num, line in enumerate(ef):
                word, *embeddingVector = line.split(' ')
                lookup[word] = line_num
                glove[line_num] = [float(i) for i in embeddingVector]
    return glove, lookup

def main(vocab_size=100_000, num_hashtags=50, test_file = "../data/twitter-nlp/json/cache-0.json",\
    num_hashtags_print=0):
    glove_file="../data/twitter-nlp/embeddings/glove.twitter.27B.25d.txt"
    embedding_dim = 25
    glove_size = 1193514

    json = open(test_file)
    twitter_parse = TwitterJSONParse(json, vocab_size, show_progress=False)
    twitter_parse.process_tweets()
    tweets, hashtags = list(Tweet.set()), list(Hashtag.set())
    print(f"Num tweets: {len(tweets)}, Num unique hashtags: {len(hashtags)}.")

    sorted_hts = sorted((hashtags))[-1::-1]
    if num_hashtags_print > 0:
        for ht in sorted_hts[:num_hashtags_print]:
            print(ht, ht.popularity)
    print(f"Loading GloVe embeddings ({embedding_dim}).")
    glove25, glove_lookup = loadGlove(glove_file, glove_size, embedding_dim)
    #print("Initializing BaselineModel")
    #glove_params = (glove25, glove_size, glove_lookup)
    #training_params = (vocab_size, embedding_dim, num_hashtags)
    #blm = bm.BaselineModel(tweets, sorted_hts, glove_params, training_params)

    print("Encoding tweets for VAE.")
    #test encode some tweets
    max_word_count = 128
    batch_size = 20
    num_tweets = 20
    #TODO encapsulate this into its own class
    _x = [[glove_lookup[word] for word in tweet.text.split()] for tweet in tweets[:num_tweets]]
    x = np.zeros((batch_size, max_word_count))
    for row in range(num_tweets):
        for i in range(len(_x[row])):
            x[row, i] = _x[row][i]
    x = tf.constant(x, dtype=tf.float32)

    vae = VAE(word_count=max_word_count, batch_size=batch_size)
    print("Building VAE.")
    vae.build_graph(x)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--vocab', type=int, default=100000)
    parser.add_argument('-p', '--numprint', type=int, default=0)
    parser.add_argument('-a', '--hashtags', type=int, default=50)
    parser.add_argument('-t', '--testfile', default='../data/twitter-nlp/json/cache-0.json')
    args = parser.parse_args()
    main(vocab_size=args.vocab, num_hashtags=args.hashtags, test_file=args.testfile, num_hashtags_print=args.numprint)
