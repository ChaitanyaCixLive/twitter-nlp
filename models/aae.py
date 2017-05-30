# Shishir Tandale
# Inspired by arxiv:1702.02390 (Semeniuta, Severyn, and Barth, 2017)

import tensorflow as tf, numpy as np

class AAE_Model(object):
    def __init__(self, encoded_tweets, character_count=160, num_layers=5, kernel_size=4,
                scope='twitternlp_aae', batch_size=20, latent_size=280, epochs=1):
        self.num_layers = num_layers
        self.character_count = character_count
        #size convolution stack reduces data to
        self.conv_reduction_size = int(self.character_count/2**self.num_layers)
        self.latent_size = latent_size
        #half of our latent vector is z, the other half is sigma
        self.z_size = int(self.latent_size/2)
        #conv stack filters
        self.filters = [128, 256, 512, 512, 512][:num_layers]
        self.decode_filters = self.filters[-1::-1]
        self.kernel_size = kernel_size
        self.ckpt_file_name = "./aae_model.ckpt"
        self.scope = scope
        self.batch_size = batch_size
        self.epochs = epochs
        #number of times critic is trained before generator
        self.n_critic = 5

        self.learning_rate = 0.0001
        self.encoder_aux_decoder_loss = 0.02
        self.critic_lambda = 8
        self.encoded_tweets = encoded_tweets
        #split encoded tweets up by character_count, then batch_size. should result in num_batches of arrays
        #calculate total number of even-sized batches
        self.num_batches = int(len(self.encoded_tweets)/(self.character_count*self.batch_size))
        self.encoded_tweets = self.encoded_tweets[:self.num_batches*self.batch_size*self.character_count]
        self.encoded_tweets = self.encoded_tweets.reshape(self.num_batches, self.batch_size, self.character_count)
        print(f"Loaded {len(self.encoded_tweets)} batches of input into VAE.")

        self.x = tf.placeholder(tf.float32, shape=[None, self.character_count])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_size])

        #builds network ops, training ops
        self.build_network()

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def build_network(self):
        #network ops
        #building blocks
        self._encoder_net = self._encoder(self.x, reuse=False)
        self._decoder_net = self._decoder(self.z, reuse=False)
        self._critic_net = self._critic(self.z, reuse=False)
        #combinations of building blocks
        self.encoder_op = self.input_encoder(self.x)
        self.enc_dec_op = self.encoder_decoder()
        self.enc_critic_op = self.encoder_critic()
        self.random_critic_op = self.random_critic()
        #collect model variables
        self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
        self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "critic")
        #training ops
        encoder_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
        generator_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)
        critic_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.5, 0.9)

        #TODO improve generation loss (use another discriminator/ GAN architecture?)
        self.generation_loss_op = self.generation_loss()

        #factor in poor generation into our encoder loss
        self.encoder_loss_op = self.encoder_loss() + self.encoder_aux_decoder_loss*self.generation_loss_op

        #calculate gradient penalty to enhance our critic loss
        z_real = tf.random_normal([self.batch_size, self.z_size], mean=0, stddev=1)
        z_fake = self.encoder_op
        eps = tf.random_uniform([1], 0, 1)
        z_hat = tf.add(tf.multiply(eps, z_real), tf.multiply(1-eps, z_fake))
        grads = tf.norm(tf.gradients(self._critic(z_hat), [z_hat])[0], 2)
        self.critic_gradient_penalty = tf.square(tf.add(grads, -1))
        self.critic_loss_op = self.critic_loss() + self.critic_lambda*self.critic_gradient_penalty

        self.train_encoder_op = encoder_optimizer.minimize(self.encoder_loss_op, var_list=self.enc_vars)
        self.train_generator_op = generator_optimizer.minimize(self.generation_loss_op, var_list=self.dec_vars)
        self.train_critic_op = critic_optimizer.minimize(self.critic_loss_op, var_list=self.critic_vars)


    def encoder_loss(self):
        return tf.reduce_mean(-self._critic(self.encoder_op))
    def critic_loss(self):
        #original critic loss
        z_real = tf.random_normal([self.batch_size, self.z_size], mean=0, stddev=1)
        z_fake = self.encoder_op
        w_real = self._critic(z_real)
        w_fake = self._critic(z_fake)
        critic_loss = tf.add(w_fake, -w_real)
        return tf.reduce_mean(critic_loss)
    def generation_loss(self):
        _x = self.enc_dec_op
        x = self.x
        #mse
        reconstruction_loss = tf.reduce_mean(tf.square(tf.add(x, -_x)))
        return tf.reduce_mean(reconstruction_loss)
    def input_encoder(self, x):
        self.mu, self.log_sigma_sq = self._encoder(x)
        #eps = tf.random_normal((self.batch_size, self.z_size), 1, 0.001, dtype=tf.float32)
        #z = mu + sqrt(exp(log_sigma_sq)) * epsilon
        return tf.add(self.mu, (tf.sqrt(tf.exp(self.log_sigma_sq))))
    def encoder_decoder(self):
        #single pass through autoencoder
        return self._decoder(self.encoder_op)
    def encoder_critic(self):
        #encode x, pass through critic
        return self._critic(self.encoder_op)
    def random_critic(self):
        #pass random normal through critic
        z = tf.random_normal([self.batch_size, self.z_size], mean=0, stddev=1)
        return self._critic(z)

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
            else:
                raise IndexError

        from time import time
        print(f"Training AAE for {self.epochs} epochs, {self.num_batches} batches each.")
        with tf.Session().as_default() as sess:
            try:
                sess.run(self.init_op)
                for e in range(self.epochs):
                    batch_pointer = 0
                    for b in range(self.num_batches):
                        n_critic = 100 if b+e*self.num_batches < 25 else self.n_critic
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
            #net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            net = tf.layers.dense(
                inputs=net,
                units=int(2*self.latent_size/3),
                name="dense_critic_2")
            #net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            net = tf.layers.dense(
                inputs=net,
                units=int(self.latent_size/3),
                name="dense_critic_3")
            #net = tf.layers.batch_normalization(net)
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
                    kernel_size=7 if layer==0 else self.kernel_size,
                    strides=2,
                    padding="same",
                    name=f"conv{layer+1}")
                #net = tf.layers.batch_normalization(net)
                net = tf.nn.relu(net)
            #flatten results of convolution stack, feed through dense layer
            #cannot get shape of last_layer automatically, so we rely on obj vars
            num_nodes = (self.conv_reduction_size)*(self.filters[-1])
            net = tf.reshape(net, [-1, num_nodes])
            #returns output of linear layer, chopped into two pieces -- mu and log sigma sq
            #each is z_size big, which is half of our latent vector size.
            mu = tf.tanh(tf.layers.dense(inputs=net, units=self.z_size, name="encode_mu"))
            log_sigma_sq = tf.tanh(tf.layers.dense(inputs=net, units=self.z_size, name="encode_logsigmasq"))
            return mu, log_sigma_sq

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
                net = tf.nn.relu(net)
            #reduce dimensionality and reshape to sentence
            net = conv_t(net, self.num_layers, filters=1, strides=1)
            net = tf.reshape(net, [-1, self.character_count])
            return net
