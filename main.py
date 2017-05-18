# Shishir Tandale

import numpy as np
import json

from utils.twitter.objects import Tweet, User, Hashtag
from utils.twitter.json import Twitter_JSON_Parse
from utils.twitter.tokenizer import Tweet_Tokenizer
from utils.twitter.tweepy import Tweepy_Client, Tweepy_Stream_Saver
from utils.embeddings import Embedder
from models.baseline import Baseline_Model
from models.aae import AAE_Model
from models.lstm import LSTM_Model

def load_glove(glove_file, glove_shape):
    from collections import defaultdict
    lookup = defaultdict(lambda: -1, dict())
    glove = np.zeros(glove_shape)
    with open(glove_file, "r") as ef:
        for line_num, line in enumerate(ef):
            word, *embedding_vector = line.split(' ')
            lookup[word] = line_num
            glove[line_num] = [float(i) for i in embedding_vector]
    return glove, lookup

def main(num_tweets, num_hashtags, test_file, num_hashtags_print, username, epochs, batch_size):
    #Load glove embeddings into Embedder
    glove_size = 1193514
    embedding_dim = 25
    glove_shape = (glove_size, embedding_dim)
    glove_file="../data/twitter-nlp/embeddings/glove.twitter.27B.25d.txt"
    print(f"Loading GloVe embeddings ({embedding_dim}).")
    glove_params = load_glove(glove_file, glove_shape)
    embedder = Embedder(glove_params)

    #load json from test_file, parse with Twitter_JSON_Parse
    #json = open(test_file).readlines()[:num_tweets]
    #print(f"Downloading {num_tweets} new tweets.")
    #client = Tweepy_Client()
    #tss = Tweepy_Stream_Saver(max_count = num_tweets)
    #client.start_stream("the", tss)
    #client.retrieve(username, num_tweets)
    json_text = open(test_file).readlines()[:num_tweets]
    #print(json)

    #parse json
    twitter_parse = Twitter_JSON_Parse(json_text)
    tweets, hashtags = list(Tweet.set), list(Hashtag.set)
    print(f"Num tweets: {len(tweets)}, Num unique hashtags: {len(hashtags)}.")

    #package params and initialize baseline model
    #print("Initializing Baseline Model.")
    #blm = Baseline_Model(tweets, hashtags, embedding_dim, epochs, batch_size)
    #blm.create_embeddings(embedder)
    #blm.train_model()
    #sentences = [t.orig_text for t in tweets[:6]]
    sentences = [
        "Do Offred and the Commander actually know the rules of Scrabble?",
        "Sunday shows struggle to book anyone willing to speak on behalf of Donald Trump",
        "The Democrats, without a leader, have become the party of obstruction.They are only interested in themselves and not in what's best for U.S.",
        "TRUMP,  when will you understand that I am not paying for that fucken wall. Be clear with US tax payers. They will pay for it."
    ]
    #blm.predict(sentences, embedder)


    print("Encoding tweets for AAE.")
    #test encode some tweets
    character_count = 160
    batch_size = batch_size
    char_map, tweet_str = Tweet_Tokenizer.build_character_map(
                                    tweets, batch_size=character_count)
    tweet_np = np.array(tweet_str)
    #build model and model graph
    aae = AAE_Model(tweet_np, character_count=character_count, batch_size=batch_size, epochs=epochs)
    aae.train()
    for line in aae.generate_random_sample(15):
        print(Tweet_Tokenizer.parse_character_map(line, char_map))
        print()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--numtweets', type=int, default=100_000)
    parser.add_argument('--username', default='realDonaldTrump')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batchsize', type=int, default=25)
    parser.add_argument('-p', '--numprint', type=int, default=0)
    parser.add_argument('-a', '--hashtags', type=int, default=50)
    parser.add_argument('-t', '--testfile', default='../data/twitter-nlp/json/cache-0.json')
    args = parser.parse_args()
    main(
        args.numtweets,
        args.hashtags,
        args.testfile,
        args.numprint,
        args.username,
        args.epochs,
        args.batchsize
    )
