# Shishir Tandale
import tensorflow as tf, numpy as np, os
from tensorflow.contrib.tensorboard.plugins import projector

class BaselineModel(object):
    def __init__(self, tweets, hashtags, hashtagMap, gloveEmbeddings, gloveSize, wordMap, vocabSize, embeddingDim, numHashtags):
        self.tweets = tweets
        self.hashtags = hashtags
        self.hashtagMap = hashtagMap
        self.numHashtags = numHashtags
        self.embeddingDim = embeddingDim
        self.gloveSize = gloveSize
        self.wordMap = wordMap
        self.LOG_DIR = "log"

        print("Converting Glove embeddings to tensor")
        self.wordEmbeddings = tf.Variable(tf.constant(0.0, shape=[gloveSize, embeddingDim]), trainable=False, name="wordEmbeddings")
        glovePlaceholder = tf.placeholder(tf.float32, [gloveSize, embeddingDim])
        gloveInit = self.wordEmbeddings.assign(glovePlaceholder)

        self.sess = tf.Session()
        with self.sess.as_default():
            self.sess.run(gloveInit, feed_dict={glovePlaceholder: gloveEmbeddings})
            print("Embedding {} most common hashtags".format(self.numHashtags))
            hashtag_embed_vector = [self.trainHashtag(h) for h in self.hashtags[:self.numHashtags]]
            self.hashtagEmbeddings = tf.stack(hashtag_embed_vector).eval()

            # self.saveEmbeddings(self.sess)

    def saveEmbeddings(self, session):
        print("Saving embeddings for visualization")
        self.saver = tf.train.Saver()
        self.saver.save(session, os.path.join(self.LOG_DIR, "model.ckpt"))

    def tweetEmbedding(self, tweet):
        if tweet.embedding == None:
            word_ids = [self.wordMap[word] for word in tweet.text.split() if word in self.wordMap.keys()]
            if word_ids == []:
                return tf.zeros((25))
            embedded_words = tf.nn.embedding_lookup(self.wordEmbeddings, word_ids)
            # just average together glove embeddings
            # TODO: improve with TF_IDF: http://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence
            tweet.embedding = tf.reduce_mean(embedded_words, 0)
        return tweet.embedding

    def trainHashtag(self, hashtag):
        tweets = self.hashtagMap[hashtag]
        tweet_embeddings = tf.stack([self.tweetEmbedding(t) for t in tweets])
        return tf.reduce_mean(tweet_embeddings, 0)
    #feed embeddings through covnet and feed forward
    #backprop to predict hashtags
