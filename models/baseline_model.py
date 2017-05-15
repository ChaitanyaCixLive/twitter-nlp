# Shishir Tandale
import tensorflow as tf, numpy as np
import pickle, sys, os

from sklearn.neighbors import BallTree

from utils.progress import Progress
from utils.twitter_json import Hashtag, Tweet, User
from utils.embeddings import Embedder

class Baseline_Model(object):
    def __init__(self, tweets, hashtags, embedding_dim):
        self.embedding_dim = embedding_dim
        self.tweets = tweets
        self.hashtags = hashtags
        self.num_tweets = len(tweets)
        self.embedding_file_name = "embedded_hashtags_tweets.p"
        self.ckpt_file_name = "./baseline_model.ckpt"
        self.hidden_layer_nodes = self.embedding_dim*2
        self.batch_size = 25
        self.learning_rate = 0.001
        self.sess = tf.Session()

        #input and output vars
        self.tweet_batch = tf.placeholder(tf.float32, shape=[None, self.embedding_dim])
        self.hashtag_batch = tf.placeholder(tf.float32, shape=[None, self.embedding_dim])
        #build baseline graph
        self.nn_ht_prediction = self.inference(self.tweet_batch, num_hidden=2)
        self.nn_ht_loss = self.loss(self.nn_ht_prediction, self.hashtag_batch)
        self.nn_train_op = self.train(self.nn_ht_loss, self.learning_rate)
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def create_embeddings(self, embedder):
        #embed tweets and hashtags
        embedder.embed_tweets_hashtags(self.tweets, self.hashtags)
        self.tweet_embeddings, self.tweet_cache = [], []
        self.hashtag_embeddings, self.hashtag_cache = [], []
        for hashtag in self.hashtags:
            for tweet in hashtag.parent_tweets:
                self.hashtag_embeddings.append(hashtag.embedding); self.tweet_embeddings.append(tweet.embedding)
                self.hashtag_cache.append(hashtag); self.tweet_cache.append(tweet)
        #build BallTree to allow for nearest neighbor searches on hashtag embeddings
        self.hashtag_search_tree = BallTree(self.hashtag_embeddings, leaf_size=40)
        #save calculated embeddings for prediction
        sys.setrecursionlimit(10_000)
        pickle.dump(
            [self.hashtag_cache, self.hashtag_search_tree],
            open(self.embedding_file_name, "wb"))
        print(f"Saved calculated embeddings to {self.embedding_file_name}.")
        self.embeddings_ready = True

    def train_model(self, epochs=5):
        from time import time
        tweets, hashtags = self.tweet_embeddings, self.hashtag_embeddings
        batches = int(len(tweets)/self.batch_size)
        def next_batch(pointer):
            if pointer+self.batch_size < len(tweets):
                tweet_batch = tweets[pointer:pointer+self.batch_size]
                hashtag_batch = hashtags[pointer:pointer+self.batch_size]
                pointer += self.batch_size
                return tweet_batch, hashtag_batch
            else:
                raise tf.errors.OutOfRangeError
        print(f"Training baseline model for {epochs+1} epochs ({batches} batches each)")
        with self.sess.as_default():
            self.sess.run(self.init_op)
            try:
                for e in range(epochs):
                    pointer = 0 #reset batch pointer
                    for b in range(batches*2):
                        start_time = time()
                        tweet_batch, hashtag_batch = next_batch(pointer)
                        _, loss_value = self.sess.run(
                            self.nn_train_op,
                            feed_dict = {
                                self.tweet_batch: tweet_batch,
                                self.hashtag_batch: hashtag_batch
                            })
                        duration = time() - start_time
                        if b % 200 == 0:
                            print(f"Epoch {e+1}, Batch {b}: loss={loss_value}, {duration} sec")
                    #between epochs, save training data
                    save_path = self.saver.save(self.sess, self.ckpt_file_name)
                    print(f"Model saved in file {save_path}")
            except KeyboardInterrupt:
                print("Training stopped by user.")
            except tf.errors.OutOfRangeError:
                print("Ran out of training data.")

    def predict(self, sentences, embedder, num=3):
        hashtags, tree = pickle.load(open(self.embedding_file_name, "rb"))
        embedding = embedder.embed_sentences(sentences)
        y = None
        with self.sess.as_default():
            self.saver.restore(self.sess, self.ckpt_file_name)
            y = self.sess.run(
                self.nn_ht_prediction,
                feed_dict = {self.tweet_batch: embedding})
        for i, sentence in enumerate(sentences):
            #todo query all at once for speed
            dist, ind = tree.query(y[i].reshape(1, -1), k=num)
            predicted_hashtags = [f"#{str(hashtags[idh])}: {dist[0][idi]}" for idi,idh in enumerate(ind[0])]
            print(f"""
{sentence}
{predicted_hashtags}
            """)

    def inference(self, tweets, num_hidden=2):
        last_layer = tweets
        for n in range(1, num_hidden):
            with tf.name_scope(f'hidden{n}'):
                input_size = self.embedding_dim if n==1 else self.hidden_layer_nodes
                output_size = self.hidden_layer_nodes
                weights = tf.Variable(tf.truncated_normal([input_size, output_size]), name='weights')
                biases = tf.Variable(tf.zeros([self.hidden_layer_nodes]), name='biases')
            last_layer = tf.nn.relu(tf.matmul(last_layer, weights) + biases)
        with tf.name_scope('softmax'):
            weights = tf.Variable(tf.truncated_normal([self.hidden_layer_nodes, self.embedding_dim]), name='weights')
            biases = tf.Variable(tf.zeros([self.embedding_dim]), name='biases')
        prediction = tf.matmul(last_layer, weights) + biases
        print(f"Built model graph with {num_hidden} layers.")
        return prediction

    def loss(self, prediction, actual):
        # cosine similarity (euclidean distance)
        return tf.nn.l2_loss(tf.subtract(actual, prediction), name="l2_loss")

    def train(self, loss, learning_rate):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_operation = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        return train_operation, loss
