import tensorflow as tf

class Embedder(object):

    def __init__(self, tweets, hashtags, glove_embeddings, glove_map, training_params, glove_size=25):
        (vocab_size, embedding_dim, num_hashtags) = training_params

        self.tweets = tweets
        self.hashtags = hashtags
        self.num_hashtags = num_hashtags
        self.num_tweets = len(tweets)
        self.embedding_dim = embedding_dim
        self.glove_size = glove_size
        self.glove_embeddings = glove_embeddings
        self.word_map = glove_map
        self.batch_size = 16
        self.learning_rate = 0.001
        self.iter = 10000

    def embed(self, file_path='/tmp/twitternlp.embeddings'):
        with tf.Session() as sess:
            self.tweet_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_tweets, self.embedding_dim]), trainable=False, name="tweet_embeddings")
            self.hashtag_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_hashtags, self.embedding_dim]), trainable=False, name="hashtag_embeddings")
            tweetinput_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.embedding_dim))
            hashtag_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.embedding_dim))

            print("Embedding {} most common hashtags".format(self.num_hashtags))
            self.tweet_embeddings = tf.stack([self.tweet_embedding(t, self.glove_embeddings) for t in self.tweets[:self.num_tweets]])
            self.hashtag_embeddings = tf.stack([self.train_hashtag(h, self.glove_embeddings) for h in self.hashtags[:self.num_hashtags]])

    def tweet_embedding(self, tweet, word_embeddings):
        if tweet.embedding is None:
            word_ids = [self.word_map[word] for word in tweet.text.split() if word in self.word_map.keys()]
            if word_ids == []:
                return tf.zeros((self.embedding_dim))
            embedded_words = tf.nn.embedding_lookup(word_embeddings, word_ids)
            # just average together glove embeddings
            # TODO: improve with TF_IDF: http://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence
            tweet.embedding = tf.reduce_mean(embedded_words, 0)
        return tweet.embedding
    def train_hashtag(self, hashtag, word_embeddings):
        if hashtag.embedding is None:
            tweets = hashtag.parent_tweets
            tweet_embeddings = tf.stack([self.tweet_embedding(t, word_embeddings) for t in tweets])
            hashtag.embedding = tf.reduce_mean(tweet_embeddings, 0)
        return hashtag.embedding
