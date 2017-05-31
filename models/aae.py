# Shishir Tandale
# Inspired by arxiv:1702.02390 (Semeniuta, Severyn, and Barth, 2017)

import tensorflow as tf, numpy as np

class AAE_Model(object):
    def __init__(self, encoded_tweets, character_count=160, num_layers=5, kernel_size=5,
                scope='twitternlp_aae', batch_size=30, latent_size=200, epochs=1,
                ckpt_file_name="./aae_model.ckpt"):
        self.num_layers = num_layers
        self.character_count = character_count
        self.encoded_tweets = encoded_tweets
        self.kernel_size = kernel_size
        self.ckpt_file_name = ckpt_file_name
        self.scope = scope
        self.batch_size = batch_size
        self.epochs = epochs
        #cut encoded_tweets to size (to a multiple of batch_size)
        self.num_batches = int(len(self.encoded_tweets)/(self.character_count*self.batch_size))
        self.encoded_tweets = self.encoded_tweets[:self.num_batches*self.batch_size*self.character_count]
        self.encoded_tweets = self.encoded_tweets.reshape(self.num_batches, self.batch_size, self.character_count)
        print(f"Loaded {len(self.encoded_tweets)} batches of input into VAE.")
        #calculate some values ahead of time for convenience
        self.conv_reduction_size = int(self.character_count/2**self.num_layers)
        self.latent_size = latent_size
        self.z_size = int(self.latent_size/2)
        self.filters = [128, 256, 512, 512, 512][:num_layers]
        self.decode_filters = self.filters[-1::-1]

        #number of times critic is trained before generator
        self.n_critic = 5
        #trade off between encoder listening to critic or decoder for feedback
        self.encoder_aux_decoder_loss = 0.5
        #critic gradient penalty multiplier (enforces 1-Lipschitz constraint on critic)
        self.critic_lambda = 10
        self.learning_rate = 0.0001 #from gradient penalty paper

        self.x = tf.placeholder(tf.float32, shape=[None, self.character_count])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_size])

        #builds network ops, loss ops, training ops
        self.build_network()
        self.saver = tf.train.Saver()

    def build_network(self):
        #network ops
        #building blocks
        self.random_op = tf.random_normal([self.batch_size, self.z_size], mean=0, stddev=1)
        #initialize all networks
        self._encoder_net = self._encoder(self.x, reuse=False)
        self._decoder_net = self._decoder(self.z, reuse=False)
        self._critic_net = self._critic(self.z, reuse=False)
        #combinations of building blocks
        self.encoder_op = self._encoder(self.x)
        self.enc_dec_op = self._decoder(self.encoder_op)
        self.enc_critic_op = self._critic(self.encoder_op)
        self.random_critic_op = self._critic(self.random_op)
        #collect model variables
        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
        self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "critic")

        #loss and training ops
        #calculate losses
        self.generation_loss_op = self.generation_loss()
        self.encoder_loss_op = self.encoder_loss()
        self.critic_loss_op = self.critic_loss()
        #calculate gradient penalty to enforce 1-Lipschitz constraint on critic
        z_real = tf.random_normal([self.batch_size, self.z_size], mean=0, stddev=1)
        z_fake = self.encoder_op
        eps = tf.random_uniform([1], 0, 1)
        z_hat = tf.add(tf.multiply(eps, z_real), tf.multiply(1-eps, z_fake))
        grads = tf.norm(tf.gradients(self._critic(z_hat), [z_hat])[0], 2)
        self.critic_gradient_penalty = self.critic_lambda * (tf.square(tf.add(grads, -1)))

        #training ops
        #helper func to interpolate linearly between two elements in a pair
        def interpolate(pair, alpha):
            return pair[0]*alpha + pair[1]*(1-alpha)
        encoder_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
        generator_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
        critic_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
        self.train_encoder_op = encoder_optimizer.minimize(
                interpolate((self.generation_loss_op, self.encoder_loss_op), self.encoder_aux_decoder_loss),
                var_list=self.enc_vars)
        self.train_critic_op = critic_optimizer.minimize(
                self.critic_loss_op + self.critic_gradient_penalty,
                var_list=self.critic_vars)
        #TODO improve generation loss (use another discriminator/ GAN architecture?)
        self.train_generator_op = generator_optimizer.minimize(self.generation_loss_op, var_list=self.dec_vars)

        self.init_op = tf.global_variables_initializer()

    def encoder_loss(self):
        return tf.reduce_mean(-self._critic(self.encoder_op))
    def critic_loss(self):
        #original critic loss
        w_real = self.random_critic_op
        w_fake = self.enc_critic_op
        critic_loss = tf.add(w_fake, -w_real)
        return tf.reduce_mean(critic_loss)
    def generation_loss(self):
        x = self.x
        _x = self.enc_dec_op
        reconstruction_loss = tf.reduce_mean(tf.square(tf.add(x, -_x))) #mse
        return reconstruction_loss

    def mean_only_batch_norm(self, net):
        mu, sigma = tf.nn.moments(net, axes=[0])
        sigma = tf.ones(sigma.get_shape().as_list()) #replace sigma with all ones
        return tf.nn.batch_normalization(net, mu, sigma, None, None, 1e-10)

    def generate_random_sample(self, num_samples=10):
        logits = np.zeros((self.batch_size, self.character_count), dtype=np.float32)
        with tf.Session().as_default() as sess:
            self.saver.restore(sess, self.ckpt_file_name)
            samples = tf.random_normal([self.batch_size, self.z_size], mean=0, stddev=1)
            logits = sess.run(self._decoder(samples))
        return logits[:num_samples]

    def train(self):
        def random_batch():
            return self.encoded_tweets[np.random.randint(0, self.num_batches)]
        def next_batch(batch_pointer):
            if batch_pointer in range(self.num_batches):
                batch = self.encoded_tweets[batch_pointer]
                batch_pointer += 1
                return batch
            else: raise IndexError
        from time import time
        print(f"Training AAE for {self.epochs} epochs, {self.num_batches} batches each.")
        with tf.Session().as_default() as sess:
            try:
                sess.run(self.init_op)
                for e in range(self.epochs):
                    batch_pointer = 0
                    for b in range(self.num_batches):
                        n_critic = 75 if b+e*self.num_batches < 30 else self.n_critic
                        #start_time = time()
                        for t in range(n_critic):
                            #_ = sess.run(self.clip_critic_weights_op)
                            _ = sess.run(self.train_critic_op,
                                feed_dict = {
                                    self.x: random_batch()
                                })

                        _, _, encoder_loss, generation_loss, critic_loss = sess.run(
                            [self.train_generator_op, self.train_encoder_op, self.encoder_loss_op, self.generation_loss_op, self.critic_loss_op],
                            feed_dict = {
                                self.x: next_batch(batch_pointer)
                            })
                        #duration = time() - start_time
                        if b % 5 == 0:
                            print(f"E({e+1}),B({b}):\tel={encoder_loss} \t gl={generation_loss} \t cl={critic_loss}")
                    save_path = self.saver.save(sess, self.ckpt_file_name)
                    print(f"Model saved in file {save_path}")
            except KeyboardInterrupt:
                print("Training stopped by user.")
                save_path = self.saver.save(sess, self.ckpt_file_name)
                print(f"Model saved in file {save_path}")
            except IndexError:
                print("Ran out of training data.")

    def _critic(self, z, reuse=True, scope="critic"):
        with tf.variable_scope(scope, reuse=reuse):
            net = tf.layers.dense(
                inputs=z,
                units=self.latent_size,
                name="dense_critic_1")
            #net = self.mean_only_batch_norm(net)
            net = tf.nn.relu(net)
            net = tf.layers.dense(
                inputs=net,
                units=int(2*self.latent_size/3),
                name="dense_critic_2")
            #net = self.mean_only_batch_norm(net)
            net = tf.nn.relu(net)
            net = tf.layers.dense(
                inputs=net,
                units=int(self.latent_size/3),
                name="dense_critic_3")
            #net = self.mean_only_batch_norm(net)
            net = tf.nn.relu(net)
            net = tf.layers.dense(
                inputs=net,
                units=1,
                name="dense_critic_output")
            return net

    def _encoder(self, x, reuse=True, scope="encoder"):
        with tf.variable_scope(scope, reuse=reuse):
            net = tf.reshape(x, [-1, self.character_count, 1])
            for layer in range(self.num_layers):
                #because strides=2, our resolution is cut in half for each layer
                net = tf.layers.conv1d(
                    inputs=net,
                    filters=self.filters[layer],
                    kernel_size=self.kernel_size,
                    strides=2,
                    padding="same",
                    name=f"conv{layer+1}")
                net = self.mean_only_batch_norm(net)
                net = tf.nn.relu(net)
            #flatten results of convolution stack, feed through dense layer
            #cannot get shape of last_layer automatically, so we rely on obj vars
            num_nodes = (self.conv_reduction_size)*(self.filters[-1])
            net = tf.reshape(net, [-1, num_nodes])
            #returns output of linear layer, chopped into two pieces -- mu and log sigma sq
            #each is z_size big, which is half of our latent vector size.
            net = tf.layers.dense(inputs=net, units=self.latent_size)
            mu = net[:, :self.z_size]
            log_sigma_sq = net[:, self.z_size:]
            #z = mu + sqrt(exp(log_sigma_sq))
            return tf.add(tf.tanh(mu), (tf.sqrt(tf.exp(tf.tanh(log_sigma_sq)))))

    def _decoder(self, z, reuse=True, scope="decoder"):
        with tf.variable_scope(scope, reuse=reuse):
            def conv_t(net, layer, filters=None, strides=2):
                return tf.layers.conv2d_transpose(
                        inputs=net,
                        filters=self.decode_filters[layer] if filters is None else filters,
                        kernel_size=self.kernel_size,
                        strides=[strides,1],
                        padding="same",
                        name=f"deconv{layer+1}")
            #convert from z to conv_reduction_size for scaling to final size
            net = tf.layers.dense(
                inputs=z,
                units=self.conv_reduction_size,
                name="dense_decode")
            #requires reshape to use transposed 2D convolutions on 1D data
            net = tf.reshape(net, [-1, self.conv_reduction_size, 1, 1])
            #net = tf.tanh(net)
            #conv_t+relu stack
            for layer in range(self.num_layers):
                net = conv_t(net, layer)
                #mean-only batch norm from arxiv:1602.07868
                net = self.mean_only_batch_norm(net)
                #crelu achieves better training performance in conv stacks
                net = tf.nn.crelu(net)
            #reduce dimensionality and reshape to sentence
            net = conv_t(net, self.num_layers, filters=1, strides=1)
            net = tf.reshape(net, [-1, self.character_count])
            return net
