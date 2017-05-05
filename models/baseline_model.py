# Shishir Tandale
import tensorflow as tf, numpy as np, os.path, time

from utils.progress import Progress
from utils.reverse_dict import Symmetric_Dictionary as sdict
from utils.twitter_json import Hashtag, Tweet, User, TwitterJSONParse

class BaselineModel(object):
    def __init__(self, tweets, hashtags, glove_params, training_params):
        (gloveEmbeddings, gloveSize, wordMap) = glove_params
        (vocabSize, embeddingDim, numHashtags) = training_params

        self.tweets = tweets
        self.hashtags = hashtags
        self.numHashtags = numHashtags
        self.numTweets = len(tweets)
        self.embeddingDim = embeddingDim
        self.hiddenLayerNodes = embeddingDim
        self.gloveSize = gloveSize
        self.wordMap = wordMap
        self.batchSize = 16
        self.learningRate = 0.001
        self.iter = 10000
        self.filenames = {'embeddings':'/tmp/twitternlp_embeddings'}

        # Create graph
        with tf.Graph().as_default():
            self.wordEmbeddings = tf.Variable(tf.constant(0.0, shape=[gloveSize, embeddingDim]), trainable=False, name="wordEmbeddings")
            glovePlaceholder = tf.placeholder(tf.float32, [gloveSize, embeddingDim])
            gloveInit = self.wordEmbeddings.assign(glovePlaceholder)

            self.tweetEmbeddings = tf.Variable(tf.constant(0.0, shape=[self.numTweets, embeddingDim]), trainable=False, name="tweetEmbeddings")
            self.hashtagEmbeddings = tf.Variable(tf.constant(0.0, shape=[numHashtags, embeddingDim]), trainable=False, name="hashtagEmbeddings")

            tweetinput_placeholder = tf.placeholder(tf.float32, shape=(self.batchSize, embeddingDim))
            hashtag_placeholder = tf.placeholder(tf.float32, shape=(self.batchSize, embeddingDim))

            nn_ht_predictions = self.inference(tweetinput_placeholder)
            nn_ht_loss = self.loss(nn_ht_predictions, hashtag_placeholder)
            nn_train_op = self.train(nn_ht_loss, self.learningRate)

            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver({'hashtagEmbeddings':self.hashtagEmbeddings, 'tweetEmbeddings':self.tweetEmbeddings})
            with tf.Session() as sess:
                sess.run(init_op)
                sess.run(gloveInit, feed_dict={glovePlaceholder: gloveEmbeddings})
                #TODO optimize, this is very slow
                print("Embedding {} most common hashtags".format(self.numHashtags))
                #calculate operation
                tweetEmbeddings = tf.stack([self.tweetEmbedding(t, i) for i, t in enumerate(self.tweets[:self.numTweets])])
                hashtagEmbeddings = tf.stack([self.trainHashtag(h, i) for i, h in enumerate(self.hashtags[:self.numHashtags])])
                #execute and store to proper tensors
                print("Evaluating embedding tensors")
                sess.run(self.hashtagEmbeddings.assign(hashtagEmbeddings))
                sess.run(self.tweetEmbeddings.assign(tweetEmbeddings))
                #save checkpoint
                save_path = saver.save(sess, self.filenames['embeddings'])
                print("Embeddings saved to {}".format(save_path))
                #save np arrays for use in other scripts
                #TODO optimize
                self.finishedHTEmbeddings = self.hashtagEmbeddings.eval(session=sess)
                self.finishedTweetEmbeddings = self.tweetEmbeddings.eval(session=sess)

    def train_steps(self):
        #TODO untested, unoptimized
        with tf.Session() as sess:
            print("Collecting tweets and hashtags for training")
            tw_ht = [(row_num, self.hashtag_tweet_map[hashtag.id], hashtag.id) for row_num, hashtag in enumerate(self.hashtags[:self.numHashtags])]
            io_pairs = []
            for row_num, tweet_ids, ht_id in tw_ht:
                ht_embed = self.finishedHTEmbeddings[row_num]
                #TODO see if there's a better way to do this
                io_pairs.extend([(self.finishedTweetEmbeddings[self.tweet_embedding_map.get(t, 0)], ht_embed) for t in tweet_ids])
            def feed_io_pairs(io_pairs, num_pairs):
                #TODO handle uneven number of pairs
                t = io_pairs[:num_pairs]
                io_pairs = io_pairs[num_pairs:]
                return t
            print("Starting training process")
            for iteration in range(self.iter):
                start_time = time.time()
                tweet_embeds, ht_embeds = zip(*feed_io_pairs(io_pairs, self.batchSize))
                tweet_embeds, ht_embeds = np.stack(tweet_embeds), np.stack(ht_embeds)
                #print(tweet_embeds, ht_embeds)
                feed_dict = {
                    tweetinput_placeholder: tweet_embeds,
                    hashtag_placeholder: ht_embeds
                }
                _, loss_value = sess.run(nn_train_op, feed_dict=feed_dict)
                duration = time.time() - start_time
                if iteration % 100 == 0:
                    print("Step {}: loss = {}, {} sec".format(iteration, loss_value, duration))

    def tweetEmbedding(self, tweet, tw_id=None):
        if tweet.embedding is None:
            word_ids = [self.wordMap[word] for word in tweet.text.split() if word in self.wordMap.keys()]
            if word_ids == []:
                return tf.zeros((self.embeddingDim))
            embedded_words = tf.nn.embedding_lookup(self.wordEmbeddings, word_ids)
            # just average together glove embeddings
            # TODO: improve with TF_IDF: http://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence
            tweet.embedding = tf.reduce_mean(embedded_words, 0)
        return tweet.embedding
    def trainHashtag(self, hashtag, ht_id=None):
        if hashtag.embedding is None:
            tweets = hashtag.parent_tweets
            tweet_embeddings = tf.stack([self.tweetEmbedding(t) for t in tweets])
            hashtag.embedding = tf.reduce_mean(tweet_embeddings, 0)
        return hashtag.embedding

    def inference(self, tweets):
        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal([self.embeddingDim, self.hiddenLayerNodes]), name='weights')
            biases = tf.Variable(tf.zeros([self.hiddenLayerNodes]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(tweets, weights) + biases)
        with tf.name_scope('hidden2'):
            weights = tf.Variable(tf.truncated_normal([self.hiddenLayerNodes, self.hiddenLayerNodes]), name='weights')
            biases = tf.Variable(tf.zeros([self.hiddenLayerNodes]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        with tf.name_scope('softmax'):
            weights = tf.Variable(tf.truncated_normal([self.hiddenLayerNodes, self.embeddingDim]), name='weights')
            biases = tf.Variable(tf.zeros([self.embeddingDim]), name='biases')
            prediction = tf.matmul(hidden2, weights) + biases
        return prediction

    def loss(self, prediction, actual):
        #euclidean distance in the correct amount of dimensions
        return tf.sqrt(tf.reduce_sum(tf.square(prediction - actual)), name='euclideanDistance')

    def train(self, loss, learningRate):
        train_opt = tf.train.GradientDescentOptimizer(learningRate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_operation = train_opt.minimize(loss, global_step=global_step)
        return train_operation, loss

if __name__ == "__main__":
    pass
