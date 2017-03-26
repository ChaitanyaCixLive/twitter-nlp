# Shishir Tandale
import tensorflow as tf, numpy as np

class BaselineModel(object):
    def __init__(self, tweets, hashtags, hashtagMap, gloveEmbeddings, gloveSize, wordMap, vocabSize, embeddingDim, numHashtags):
        self.tweets = tweets
        self.hashtags = hashtags
        self.hashtagMap = hashtagMap
        self.numHashtags = numHashtags
        self.embeddingDim = embeddingDim
        self.gloveSize = gloveSize
        self.wordMap = wordMap

        print("Converting Glove embeddings to tensor")
        self.wordEmbeddings = tf.Variable(tf.constant(0.0, shape=[gloveSize, embeddingDim]), trainable=False, name="wordEmbeddings")
        glovePlaceholder = tf.placeholder(tf.float32, [gloveSize, embeddingDim])
        gloveInit = self.wordEmbeddings.assign(glovePlaceholder)

        self.sess = tf.Session()
        with self.sess.as_default():
            #TODO fix word embeddings not being initalized properly
            self.sess.run(gloveInit, feed_dict={glovePlaceholder: gloveEmbeddings})
            print("Embedding {} most common hashtags".format(self.numHashtags))
            hashtag_embed_vector = [self.trainHashtag(h) for h in self.hashtags[:self.numHashtags]]
            self.hashtagEmbeddings = tf.stack(hashtag_embed_vector).eval()

    def tweetEmbedding(self, tweet):
        word_ids = [self.wordMap[word] for word in tweet.split() if word in self.wordMap.keys()]
        if word_ids == []:
            return tf.zeros((25))
        embedded_words = tf.nn.embedding_lookup(self.wordEmbeddings, word_ids)
        return tf.reduce_sum(embedded_words, 0)
    def trainHashtag(self, hashtag):
        tweet_embeddings = tf.stack([self.tweetEmbedding(t) for t in self.hashtagMap[hashtag]])
        return tf.reduce_sum(tweet_embeddings, 0)
    #feed embeddings through covnet and feed forward
    #backprop to predict hashtags
