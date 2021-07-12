import tensorflow as tf
import numpy as np
from src import utils
import config


class Generator(object):
    def __init__(self, n_node, n_att):
        self.n_node = n_node
        self.n_att = n_att

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.get_variable(name="node_embedding",
                                                     shape=[(n_att+n_node),config.n_emb],
                                                     initializer=tf.constant_initializer(np.concatenate((utils.read_embeddings(filename=config.pretrain_node_emb_filename_g),utils.read_embeddings(filename=config.pretrain_att_emb_filename_g)), axis=0)),
                                                     trainable=True)

            self.bias_vector = tf.Variable(tf.zeros([self.n_node + self.n_att]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.reward = tf.placeholder(tf.float32, shape=[None])

        self.all_score = tf.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)

        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)

        self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + config.lambda_gen * (
                tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding))
        self.g_updates = tf.train.AdamOptimizer(config.lr_gen).minimize(self.loss)
