# Shishir Tandale
# Inspired by arxiv:1702.02390 (Semeniuta, Severyn, and Barth, 2017)

import tensorflow as tf

class Hybrid_VAE_Model(object):
    def __init__(self, word_count=128, num_layers=4, kernel_size=3,\
                scope='twitternlp_hybridvae', batch_size=20, latent_size=20):
        self.num_layers = num_layers
        self.word_count = word_count
        self.conv_reduction_size = int(self.word_count/2**self.num_layers)
        self.latent_size = latent_size
        self.z_size = int(self.latent_size/2)
        #self.encoder_inputs = tf.placeholder(tf.float32, [None, self.word_count])
        #self.decoder_inputs = tf.placeholder(tf.float32, [None, self.latent_size])
        self.filters = [128, 256, 512, 512, 512][:num_layers]
        self.decode_filters = self.filters[-1::-1]
        self.kernel_size = kernel_size
        self.scope = scope
        self.batch_size = batch_size

    def build_graph(self, x, alpha=0.3):
        # inspired by https://github.com/andrewliao11/VAE-tensorflow main.py
        with tf.variable_scope(self.scope):
            #encode inputs
            self.mu, self.log_sigma_sq = self._encode(x)
            epsilon = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
            #z = mu + sqrt(exp(log_sigma_sq)) * epsilon
            self.z = tf.add(self.mu, tf.multiply(tf.sqrt(tf.exp(self.log_sigma_sq)), epsilon))
            #decode encoded inputs
            _x = self._decode(self.z)
            ##TODO, implement auxillary cost function from arxiv:1702.02390
            #using bernoulli likelihood from https://onlinecourses.science.psu.edu/stat504/node/27
            j_aux = -tf.reduce_sum(x*tf.log(1e-8 + _x)+(1-x)*tf.log(1e-8 + 1-_x), 1)
            #KL-divergence
            j_vae = -0.5 * tf.reduce_sum(\
                1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq), 1)
            #sum but with alpha value for KL-divergence
            j_hybrid = tf.reduce_mean(tf.add(j_aux, tf.multiply(j_vae, alpha)))
            self.cost = j_hybrid

    def _encode(self, x):
        last_layer = tf.reshape(x, [-1, self.word_count, 1])
        for layer in range(self.num_layers):
            #because strides=2, our resolution is cut in half for each layer
            _conv = tf.layers.conv1d(
                inputs=last_layer,
                filters=self.filters[layer],
                kernel_size=self.kernel_size,
                strides=2,
                padding="same",
                name=f"conv{layer+1}")
            _bn = tf.layers.batch_normalization(_conv)
            _relu = tf.nn.relu(_bn)
            #store last relu reference so we can continue building our conv stack
            last_layer = _relu
        #flatten results of convolution stack, feed through dense layer
        #cannot get shape of last_layer automatically, so we rely on obj vars
        num_nodes = (self.conv_reduction_size)*(self.filters[-1])
        _flatten = tf.reshape(last_layer, [-1, num_nodes])
        _linear = tf.layers.dense(
            inputs=_flatten,
            units=self.latent_size,
            name="dense_encode")
        #returns output of linear layer, chopped into two pieces -- mu and log sigma sq
        #each is z_size big, which is half of our latent vector size.
        #mu = _linear[:, :self.z_size]
        #log_sigma_sq = _linear[:, self.z_size:]
        mu = tf.slice(_linear, [0, 0], [self.batch_size, self.z_size])
        log_sigma_sq = tf.slice(_linear, [0, self.z_size], [self.batch_size, self.z_size])
        return mu, log_sigma_sq

    def _decode(self, z):
        #one line constructor for conv2d_transpose
        def conv_t(last_layer, layer, filters=None, strides=2):
            return tf.layers.conv2d_transpose(
                    inputs=last_layer,
                    filters=self.decode_filters[layer] if filters is None else filters,
                    kernel_size=self.kernel_size,
                    strides=[strides,1],
                    padding="same",
                    name=f"deconv{layer+1}")
        #convert from latent representation to conv_reduction_size for scaling
        _linear = tf.layers.dense(
            inputs=z,
            units=self.conv_reduction_size,
            activation=tf.nn.relu,
            name="dense_decode")
        #handy reference to aid in building our conv stack
        #requires reshape to use transposed 2D convolutions on 1D data
        last_layer = tf.reshape(_linear, [-1, self.conv_reduction_size, 1, 1])
        for layer in range(self.num_layers):
            #strides=2 so our spatial resolution is doubled each layer
            _conv_t = conv_t(last_layer, layer)
            _bn = tf.layers.batch_normalization(_conv_t)
            _relu = tf.nn.relu(_bn)
            last_layer = _relu
        #final conv with a single filter and no stride to map to the right dimensionality
        _final_conv_t = conv_t(last_layer, self.num_layers, filters=1, strides=1)
        #reshape to drop extra dimensions required for conv2d_t
        return tf.reshape(_final_conv_t, [-1, self.word_count])
