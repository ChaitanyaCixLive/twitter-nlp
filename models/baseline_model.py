# Shishir Tandale

import tensorflow as tf, time
from utils.progress import Progress
from utils.reverse_dict import Symmetric_Dictionary as sdict
from utils.twitter_json import Hashtag, Tweet, User, TwitterJSONParse
from utils.embeddings import Embedder

class Baseline_Model(object):
    def __init__(self, tweets, hashtags, glove_params, training_params):
        (glove_embeddings, glove_size, glove_map) = glove_params
        (vocab_size, embedding_dim, num_hashtags) = training_params

        self.tweets = tweets
        self.hashtags = hashtags
        self.num_hashtags = num_hashtags
        self.num_tweets = len(tweets)
        self.embedding_dim = embedding_dim
        self.hidden_layer_nodes = embedding_dim

        self.batch_size = 16
        self.learning_rate = 0.001
        self.iter = 10000
        self.filenames = {'embeddings':'/tmp/twitternlp.embeddings'}

        with tf.Session() as sess:
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[glove_size, embedding_dim]), trainable=False, name="word_embeddings")
            glove_placeholder = tf.placeholder(tf.float32, [glove_size, embedding_dim])
            glove_init = self.word_embeddings.assign(glove_placeholder)
            sess.run(glove_init, feed_dict={glove_placeholder: glove_embeddings})

        tweetinput_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, embedding_dim))
        hashtag_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, embedding_dim))

        self.tweet_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_tweets, embedding_dim]), trainable=False, name="tweet_embeddings")
        self.hashtag_embeddings = tf.Variable(tf.constant(0.0, shape=[num_hashtags, embedding_dim]), trainable=False, name="hashtag_embeddings")
        saver = tf.train.Saver({'hashtag_embeddings':self.hashtag_embeddings, 'tweet_embeddings':self.tweet_embeddings})
        self.embedder = Embedder(tweets, hashtags, glove_embeddings, glove_map, training_params)
        self.embedder.embed(self.filenames['embeddings'])
        print("Loading embedding tensors into model")
        with tf.Session() as sess, tf.Graph().as_default():
            sess.run(self.hashtag_embeddings.assign(embedder.hashtag_embeddings))
            sess.run(self.tweet_embeddings.assign(embedder.tweet_embeddings))

            nn_ht_predictions = self.inference(tweetinput_placeholder)
            nn_ht_loss = self.loss(nn_ht_predictions, hashtag_placeholder)
            nn_train_op = self.train(nn_ht_loss, self.learningRate)
            init_op = tf.global_variables_initializer()
            self.train_steps()

    def train_steps(self, sess):
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

    def inference(self, tweets, num_hidden=2, ):
        for n in xrange(1, num_hidden):
            with tf.name_scope(f'hidden{n}'):
                input_size = self.embeddingDim if n==1 else self.hiddenLayerNodes
                output_size = self.hiddenLayerNodes
                weights = tf.Variable(tf.truncated_normal([input_size, output_size]), name='weights')
                biases = tf.Variable(tf.zeros([self.hiddenLayerNodes]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(tweets, weights) + biases)
        with tf.name_scope('softmax'):
            weights = tf.Variable(tf.truncated_normal([self.hiddenLayerNodes, self.embeddingDim]), name='weights')
            biases = tf.Variable(tf.zeros([self.embeddingDim]), name='biases')
            prediction = tf.matmul(hidden2, weights) + biases
        print(f"Built model graph with {num_hidden} layers.")
        return prediction

    def loss(self, prediction, actual):
        #euclidean distance in the correct amount of dimensions
        return tf.sqrt(tf.reduce_sum(tf.square(prediction - actual)), name='euclideanDistance')

    def train(self, loss, learningRate):
        train_opt = tf.train.GradientDescentOptimizer(learningRate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_operation = train_opt.minimize(loss, global_step=global_step)
        return train_operation, loss
