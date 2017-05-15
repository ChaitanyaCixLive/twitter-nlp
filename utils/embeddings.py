# Shishir Tandale

import numpy as np
from utils.progress import Progress

class Embedder(object):
    def __init__(self, glove_params):
        (self.glove_embeddings, self.glove_lookup) = glove_params

    def embed_tweets_hashtags(self, tweets, hashtags):
        with Progress("Calculating hashtag and tweet embeddings", len(hashtags)+len(tweets)) as up:
            [up(self.tweet_embedding(t)) for t in tweets]
            [up(self.hashtag_embedding(h)) for h in hashtags]

    def embed_sentence(self, sentence):
        word_ids = [self.glove_lookup[word] for word in sentence.lower().split()]
        embedded_words = [self.glove_embeddings[i] for i in word_ids]
        embedding = np.average(embedded_words, 0)
        return embedding

    def embed_sentences(self, sentences):
        return np.stack([self.embed_sentence(s) for s in sentences], 0)

    def tweet_embedding(self, tweet):
        if tweet.embedding is None:
            # TODO: improve with TF_IDF: http://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence
            tweet.embedding = self.embed_sentence(tweet.text)
        return tweet.embedding

    def hashtag_embedding(self, hashtag):
        if hashtag.embedding is None:
            tweet_embeddings = [self.tweet_embedding(t) for t in hashtag.parent_tweets]
            hashtag.embedding = np.average(tweet_embeddings, 0)
        return hashtag.embedding
