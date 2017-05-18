# Shishir Tandale
# Inspired by arxiv:1702.02390 (Semeniuta, Severyn, and Barth, 2017)

import tensorflow as tf, numpy as np

class Hybrid_VAE_Model(object):
    def __init__(self, encoded_tweets, character_count=160, num_layers=4, kernel_size=3,
                scope='twitternlp_hybridvae', batch_size=20, latent_size=30, epochs=1, clip=0.005):
        self.num_layers = num_layers
        self.character_count = character_count
        self.conv_reduction_size = int(self.character_count/2**self.num_layers)
        self.latent_size = latent_size
        self.z_size = int(self.latent_size/2)
        self.filters = [128, 256, 512, 512, 512][:num_layers]
        self.decode_filters = self.filters[-1::-1]
        self.kernel_size = kernel_size
        self.ckpt_file_name = "./hybrid_model.ckpt"
        self.scope = scope
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_critic = 5
        self.clip = clip
        self.encoded_tweets = encoded_tweets
        self.batch_pointer = 0
        print(f"Loaded {len(self.encoded_tweets)} characters of input into VAE.")

        self.x = tf.placeholder(tf.float32, shape=[None, self.character_count])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_size])

        self.enc_dec = self.encoder_decoder(self.x, reuse=None)
        self.critic = self._wasserstein_critic(self.x, reuse=None)
        self.enc_dec_critic = self._wasserstein_critic(self.enc_dec, reuse=True)
        self.encoder = self._encode(self.x, reuse=True)
        self.decoder = self._decode(self.z, reuse=True)

        #collect model variables
        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
        self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        self.encdec_vars = self.enc_vars + self.dec_vars
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "wasserstein_critic")
        #training ops
        encdec_optimizer = tf.train.RMSPropOptimizer(0.00005)
        critic_optimizer = tf.train.RMSPropOptimizer(0.00005)
        self.encdec_loss = self.encoder_decoder_loss()
        self.critic_loss = self.critic_loss()
        self.train_encdec_op = encdec_optimizer.minimize(self.encdec_loss, var_list=self.encdec_vars)
        self.train_critic_op = critic_optimizer.minimize(self.critic_loss, var_list=self.critic_vars)
        #creates array of ops for clipping
        self.clip_critic_weights_op = []
        for var in self.critic_vars:
            self.clip_critic_weights_op.append(tf.assign(var, tf.clip_by_value(var, -self.clip, self.clip)))

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def critic_loss(self):
        wx = tf.reduce_mean(self.critic, 0)
        w_x = -tf.reduce_mean(self.enc_dec_critic, 0)
        return tf.reduce_mean(tf.add(wx, w_x))

    def encoder_decoder_loss(self):
        return -tf.reduce_mean(self.enc_dec_critic)

    def next_batch(self, increment=True, num=1):
        #TODO fix batches
        start, end = self.batch_pointer, self.batch_pointer+(self.character_count*self.batch_size)*num
        if end < len(self.encoded_tweets):
            batch = self.encoded_tweets[start:end].reshape(
                (self.batch_size, self.character_count) if num==1 else \
                (num, self.batch_size, self.character_count)
            )
            if increment:
                self.batch_pointer += (self.character_count*self.batch_size)*num
            return batch
        else:
            raise IndexError

    def random_sample(self, num_samples=10):
        samples = np.random.normal(0, 2, size=[num_samples, self.z_size])
        logits = np.zeros((num_samples, self.character_count), dtype=np.float32)
        with tf.Session().as_default() as sess:
            self.saver.restore(sess, self.ckpt_file_name)
            logits = sess.run(self.decoder, feed_dict={ self.z: samples })
        print(logits.shape)
        return logits

    def train(self):
        from time import time
        batches = int(len(self.encoded_tweets)/(self.character_count*self.batch_size))
        print(f"Training VAE for {self.epochs} epochs, {batches} batches each.")

        with tf.Session().as_default() as sess:
            try:
                sess.run(self.init_op)
                for e in range(self.epochs):
                    self.batch_pointer = 0
                    for b in range(batches-10):
                        n_critic = 70 if b+e*batches < 25 else self.n_critic
                        start_time = time()
                        critic_batches = self.next_batch(increment=False, num=n_critic)
                        for t in range(n_critic):
                            _ = sess.run(self.clip_critic_weights_op)
                            _ = sess.run(self.train_critic_op,
                                feed_dict = {
                                    self.x: critic_batches[t]
                                })
                        _, wc_loss, ed_loss = sess.run(
                            [self.train_encdec_op, tf.reduce_mean(self.critic_loss), self.encdec_loss],
                            feed_dict = {
                                self.x: self.next_batch()
                            })
                        duration = time() - start_time
                        if b % 5 == 0:
                            print(f"Epoch {e+1}, Batch {b}:\twc={wc_loss}\t\ted={ed_loss}")
                        if b % 100 == 0:
                            save_path = self.saver.save(sess, self.ckpt_file_name)
                            print(f"Model saved in file {save_path}")
            except KeyboardInterrupt:
                print("Training stopped by user.")
            except IndexError:
                print("Ran out of training data.")

    def encoder_decoder(self, x, reuse=None):
        #encode inputs
        self.mu, self.log_sigma_sq = self._encode(x, reuse=reuse)
        #epsilon = tf.random_normal((self.batch_size, self.z_size), 0, 1, dtype=tf.float32)
        #z = mu + sqrt(exp(log_sigma_sq)) * epsilon
        self.z = tf.add(self.mu, tf.sqrt(tf.exp(self.log_sigma_sq)))
        return self._decode(self.z, reuse=reuse)

    def _wasserstein_critic(self, x, reuse=True, scope="wasserstein_critic"):
        with tf.variable_scope(scope, reuse=reuse):
            last_layer = tf.reshape(x, [-1, self.character_count, 1])
            for layer in range(self.num_layers):
                #because strides=2, our resolution is cut in half for each layer
                _conv = tf.layers.conv1d(
                    inputs=last_layer,
                    filters=self.filters[layer],
                    kernel_size=self.kernel_size,
                    strides=2,
                    padding="same",
                    name=f"conv{layer+1}")
                #_bn = tf.layers.batch_normalization(_conv)
                _relu = tf.nn.relu(_conv)#_bn)
                #store last relu reference so we can continue building our conv stack
                last_layer = _relu
            #flatten results of convolution stack, feed through dense layer
            #cannot get shape of last_layer automatically, so we rely on obj vars
            num_nodes = (self.conv_reduction_size)*(self.filters[-1])
            _flatten = tf.reshape(last_layer, [-1, num_nodes])
            _linear = tf.layers.dense(
                inputs=_flatten,
                units=1,
                name="dense_encode")
            return _linear

    def _encode(self, x, reuse=True, scope="encoder"):
        with tf.variable_scope(scope, reuse=reuse):
            last_layer = tf.reshape(x, [-1, self.character_count, 1])
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

    def _decode(self, z, reuse=True, scope="decoder"):
        with tf.variable_scope(scope, reuse=reuse):
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
            return tf.reshape(_final_conv_t, [-1, self.character_count])
