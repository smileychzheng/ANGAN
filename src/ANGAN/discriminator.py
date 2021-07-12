import tensorflow as tf
import numpy as np
from src import utils
import config


class Discriminator(object):
    def __init__(self, n_node, n_att):
        self.n_node = n_node
        self.n_att = n_att

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.get_variable(name="node_embedding",shape=[(n_att + n_node), config.n_emb],
                                                    initializer=tf.constant_initializer(np.concatenate((utils.read_embeddings(filename=config.pretrain_node_emb_filename_d),
                                                                                                       utils.read_embeddings(filename=config.pretrain_att_emb_filename_d)),axis=0)),
                                                    trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node + self.n_att]))

        self.node_id = tf.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])
        self.label = tf.placeholder(tf.float32, shape=[None])

        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis * (
                tf.nn.l2_loss(self.node_neighbor_embedding) +
                tf.nn.l2_loss(self.node_embedding) +
                tf.nn.l2_loss(self.bias))

        self.d_updates = tf.train.AdamOptimizer(config.lr_dis).minimize(self.loss)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
