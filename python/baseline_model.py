# Shishir Tandale
import tensorflow as tf, numpy as np, os.path

class BaselineModel(object):
    def __init__(self, tweets, hashtags, hashtagMap, gloveEmbeddings, gloveSize, wordMap, vocabSize, embeddingDim, numHashtags):
        self.tweets = tweets
        self.hashtags = hashtags
        self.hashtagMap = hashtagMap
        self.numHashtags = numHashtags
        self.numTweets = len(tweets)
        self.embeddingDim = embeddingDim
        self.hiddenLayerNodes = embeddingDim
        self.gloveSize = gloveSize
        self.wordMap = wordMap
        self.filenames = {'embeddings':'/tmp/twitternlp_embeddings'}

        # Create graph
        with tf.Graph().as_default():
            self.wordEmbeddings = tf.Variable(tf.constant(0.0, shape=[gloveSize, embeddingDim]), trainable=False, name="wordEmbeddings")
            glovePlaceholder = tf.placeholder(tf.float32, [gloveSize, embeddingDim])
            gloveInit = self.wordEmbeddings.assign(glovePlaceholder)

            self.tweetEmbeddings = tf.Variable(tf.constant(0.0, shape=[self.numTweets, embeddingDim]), trainable=False, name="tweetEmbeddings")
            self.hashtagEmbeddings = tf.Variable(tf.constant(0.0, shape=[numHashtags, embeddingDim]), trainable=False, name="hashtagEmbeddings")

            #TODO make methods to populate feed dicts with correctly shaped batches
            #self.inferenceGraph = inference()
            #TODO test network
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver({'hashtagEmbeddings':self.hashtagEmbeddings, 'tweetEmbeddings':self.tweetEmbeddings})
            with tf.Session() as sess:
                sess.run(init_op)
                sess.run(gloveInit, feed_dict={glovePlaceholder: gloveEmbeddings})
                #calculate hashtag embeddings if needed
                try:
                    print("Attempting to load saved embeddings")
                    saver.restore(sess, self.filenames['embeddings'])
                except Exception as e:
                    print("Embedding {} most common hashtags".format(self.numHashtags))
                    #calculate operation
                    hashtagEmbeddings = tf.stack([self.trainHashtag(h) for h in self.hashtags[:self.numHashtags]])
                    tweetEmbeddings = tf.stack([self.tweetEmbedding(t) for t in self.tweets[:self.numTweets]])
                    #execute and store to proper tensors
                    sess.run(self.hashtagEmbeddings.assign(hashtagEmbeddings))
                    sess.run(self.tweetEmbeddings.assign(tweetEmbeddings))
                    #save checkpoint
                    save_path = saver.save(sess, self.filenames['embeddings'])
                    print("Embeddings saved to {}".format(save_path))
                #save np arrays for use in other scripts
                self.finishedHTEmbeddings = self.hashtagEmbeddings.eval(session=sess)
                self.finishedTweetEmbeddings = self.tweetEmbeddings.eval(session=sess)

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

    def inference(self, embeddingPlaceholder):
        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal([self.embeddingDim, self.hiddenLayerNodes]), name='weights')
            biases = tf.Variable(tf.zeros([self.hiddenLayerNodes]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(embeddingPlaceholder, weights) + biases)
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
        return tf.sqrt(tf.add(tf.square(prediction - actual)), name='euclideanDistance')

    def train(self, loss, learningRate):
        train_opt = tf.train.GradientDescentOptimizer(learningRate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_operation = train_opt.minimize(loss, global_step=global_step)
        return train_operation
