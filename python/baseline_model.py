# Shishir Tandale
import tensorflow as tf, numpy as np

class BaselineModel(object):
    def __init__(self, tweets, hashtags, hashtagMap, wordEmbeddings, wordMap, vocabSize, embeddingDim, numHashtags):
        self.tweets = tweets
        self.hashtags = hashtags
        self.hashtagMap = hashtagMap
        self.numHashtags = numHashtags
        self.wordMap = wordMap

        self.wordEmbeddings = tf.get_variable(name="wordEmbeddings", shape=[vocabSize, embeddingDim], tf.constant_initializer(wordEmbeddings), trainable=False)
        self.hashtagEmbeddings = tf.get_variable(name="hashtagEmbeddings", shape=[numHashtags, embeddingDim], tf.constant_initializer([trainHashtag(h) for h in hashtags[:numHashtags]])), trainable=False, name="hashtagEmbeddings")
    def tweetEmbedding(self, tweet):
        words = tweet.split()
        def findInTable(word):
            return self.wordMap[word] if word in self.wordMap.keys() else -1
        word_ids = tf.convert_to_tensor([findInTable(w) for w in words], dtype=tf.int32)
        embedded_words = tf.nn.embedding_lookup(self.wordEmbeddings, word_ids)
        tweet_embedding = tf.concat(0, embedded_words)
        return tweet_embedding
    def trainHashtag(self, hashtag):
        tweet_embeddings = [tweetEmbedding(t) for t in self.hashtagMap[hashtag]]
        return tf.concat(0, tweet_embeddings)
    #feed embeddings through covnet and feed forward
    #backprop to predict hashtags
