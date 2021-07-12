import os
import collections
import tqdm
import multiprocessing
import pickle
import numpy as np
import tensorflow as tf
import config
import generator
import discriminator
from src import utils
from src.evaluation.Experiment import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import scipy.io as sio
import gc
from liblinear import *


class ANGAN(object):
    def __init__(self):

        print("reading graphs...")
        self.n_node, self.uu_graph = utils.read_uu_edges(config.uu_filename)
        self.ua_graph = utils.read_ua_edges(config.ua_filename)
        self.n_att, self.au_graph = utils.read_au_edges(config.au_filename)
        self.root_nodes = [i for i in range(self.n_node)]
        self.root_attnodes = [i for i in range(self.n_att)]

        print("Nodes : %d , Attributes : %d" % (self.n_node, self.n_att))
        if config.is_trained_pretrain:
            config.pretrain_node_emb_filename_d = config.trained_node_emb_filename_d
            config.pretrain_att_emb_filename_d = config.trained_att_emb_filename_d
            config.pretrain_node_emb_filename_g = config.trained_node_emb_filename_g
            config.pretrain_att_emb_filename_g = config.trained_att_emb_filename_g

        # construct or read BFS-trees
        self.uu_trees = None
        self.ua_trees = None
        self.au_trees = None
        if os.path.isfile(config.cache_filename + "_uu.pkl"):
            print("reading uu-BFS-trees from cache...")
            pickle_file = open(config.cache_filename + "_uu.pkl", 'rb')
            self.uu_trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing uu-BFS-trees...")
            if config.multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.uu_trees = self.construct_uu_trees(self.root_nodes)
            pickle_file = open(config.cache_filename + "_uu.pkl", 'wb')
            pickle.dump(self.uu_trees, pickle_file)
            pickle_file.close()

        if os.path.isfile(config.cache_filename + "_ua.pkl"):
            print("reading ua-BFS-trees from cache...")
            pickle_file = open(config.cache_filename + "_ua.pkl", 'rb')
            self.ua_trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing ua-BFS-trees...")
            self.ua_trees = self.construct_ua_trees(self.root_nodes)
            pickle_file = open(config.cache_filename + "_ua.pkl", 'wb')
            pickle.dump(self.ua_trees, pickle_file)
            pickle_file.close()

        if os.path.isfile(config.cache_filename + "_au.pkl"):
            print("reading au-BFS-trees from cache...")
            pickle_file = open(config.cache_filename + "_au.pkl", 'rb')
            self.au_trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing au-BFS-trees...")
            self.au_trees = self.construct_au_trees(self.root_attnodes)
            pickle_file = open(config.cache_filename + "_au.pkl", 'wb')
            pickle.dump(self.au_trees, pickle_file)
            pickle_file.close()

        self.fout = open(config.result_filename, "w")

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        with tf.variable_scope("generator"):
            self.generator = generator.Generator(n_node=self.n_node, n_att=self.n_att)
        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(n_node=self.n_node, n_att=self.n_att)

        self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        self.saver = tf.train.Saver()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)
        
    def construct_ua_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.ua_trees = {}
        trees_result = pool.map(self.construct_ua_trees, new_nodes)
        for tree in trees_result:
            self.ua_trees.update(tree)

    def construct_au_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_att // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.au_trees = {}
        trees_result = pool.map(self.construct_au_trees, new_nodes)
        for tree in trees_result:
            self.au_trees.update(tree)

    def construct_uu_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.uu_graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def construct_ua_trees(self, nodes):
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][self.n_att] = []
            used_nodes = set()
            queue = collections.deque([root])

            cur_node = queue.popleft()
            used_nodes.add(self.n_att)
            for sub_node in self.ua_graph[cur_node]:
                if sub_node not in used_nodes:
                    trees[root][self.n_att].append(sub_node)
                    trees[root][sub_node] = [self.n_att]
                    queue.append(sub_node)
                    used_nodes.add(sub_node)

            while len(queue) > 0:
                cur_node = queue.popleft()
                for sub_node in self.au_graph[cur_node]:
                    for sub_att_node in self.ua_graph[sub_node]:
                        if sub_att_node not in used_nodes:
                            trees[root][cur_node].append(sub_att_node)
                            trees[root][sub_att_node] = [cur_node]
                            queue.append(sub_att_node)
                            used_nodes.add(sub_att_node)
        return trees

    def construct_au_trees(self, nodes):
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][self.n_node] = []
            used_nodes = set()
            queue = collections.deque([root])

            cur_node = queue.popleft()
            used_nodes.add(self.n_node)
            for sub_node in self.au_graph[cur_node]:
                if sub_node not in used_nodes:
                    trees[root][self.n_node].append(sub_node)
                    trees[root][sub_node] = [self.n_node]
                    queue.append(sub_node)
                    used_nodes.add(sub_node)

            while len(queue) > 0:
                cur_node = queue.popleft()
                for sub_att_node in self.ua_graph[cur_node]:
                    for sub_node in self.au_graph[sub_att_node]:
                        if sub_node not in used_nodes:
                            trees[root][cur_node].append(sub_node)
                            trees[root][sub_node] = [cur_node]
                            queue.append(sub_node)
                            used_nodes.add(sub_node)
        return trees


    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                pos = self.uu_graph[i]
                neg, _ = self.uu_sample(i, self.uu_trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))
                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
                pos = self.ua_graph[i]
                neg, _ = self.ua_sample(i, self.ua_trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    pos = [i+self.n_node for i in pos]
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))
                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        for i in self.root_attnodes:
            if np.random.rand() < config.update_ratio:
                pos = self.au_graph[i]
                neg, _ = self.au_sample(i, self.au_trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i+self.n_node] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))
                    # negative samples
                    center_nodes.extend([i+self.n_node] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels


    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        paths = []
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                _, paths_from_i = self.uu_sample(i, self.uu_trees[i], config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
                _, paths_from_i = self.ua_sample(i, self.ua_trees[i], config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        for i in self.root_attnodes:
            if np.random.rand() < config.update_ratio:
                _, paths_from_i = self.au_sample(i, self.au_trees[i], config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2, reward

    def prepare_data_for_all(self):
        center_nodes = []
        neighbor_nodes = []
        labels = []
        paths = []
        node_1 = []
        node_2 = []
        all_score = self.sess.run(self.generator.all_score)
        for i in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                pos = self.uu_graph[i]
                neg, paths_from_i = self.uu_sample(i, self.uu_trees[i], all_score)
                n_pos = len(pos)
                if n_pos != 0 and len(neg) >= n_pos:
                    multip = len(neg)//n_pos
                    n_pos = n_pos * multip
                    pos = pos * multip
                    # positive samples
                    center_nodes.extend([i] * n_pos)
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * n_pos)
                    # negative samples
                    center_nodes.extend([i] * n_pos)
                    neighbor_nodes.extend(neg[:n_pos])
                    labels.extend([0] * n_pos)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
                pos = self.ua_graph[i]
                neg, paths_from_i = self.ua_sample(i, self.ua_trees[i], all_score)
                n_pos = len(pos)
                if n_pos != 0 and len(neg) >= n_pos:
                    multip = len(neg) // n_pos
                    n_pos = n_pos * multip
                    pos = pos * multip
                    # positive samples
                    pos = [i+self.n_node for i in pos]
                    center_nodes.extend([i] * n_pos)
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * n_pos)
                    # negative samples
                    center_nodes.extend([i] * n_pos)
                    neighbor_nodes.extend(neg[:n_pos])
                    labels.extend([0] * n_pos)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        for i in self.root_attnodes:
            if np.random.rand() < config.update_ratio:
                pos = self.au_graph[i]
                neg, paths_from_i = self.au_sample(i, self.au_trees[i], all_score)
                n_pos = len(pos)
                if n_pos != 0 and len(neg) >= n_pos:
                    multip = len(neg) // n_pos
                    n_pos = n_pos * multip
                    pos = pos * multip
                    # positive samples
                    center_nodes.extend([i+self.n_node] * n_pos)
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * n_pos)
                    # negative samples
                    center_nodes.extend([i+self.n_node] * n_pos)
                    neighbor_nodes.extend(neg[:n_pos])
                    labels.extend([0] * n_pos)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return center_nodes, neighbor_nodes, labels, node_1, node_2, reward



    def uu_sample(self, root, tree, all_score):
        """ sample nodes from BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        samples = []
        paths = []
        n = 0
        while n < config.n_uu_sample_gen:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None

                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    if previous_node != root:
                        samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    def ua_sample(self, u_node, tree, all_score):
        samples = []
        paths = []
        n = 0
        root = self.n_att
        while n < config.n_ua_sample_gen:
            current_node = root
            previous_node = -1
            paths.append([])
            paths[n].append(u_node)
            while True:
                node_neighbor = tree[current_node]
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                node_neighbor_tab = []
                for i in node_neighbor:
                    node_neighbor_tab.append(u_node if i == root else i+self.n_node)
                relevance_probability = all_score[u_node, node_neighbor_tab] if current_node == root else all_score[(current_node+self.n_node), node_neighbor_tab]
                relevance_probability = utils.softmax(relevance_probability)
                next = np.random.choice(range(len(node_neighbor)), size=1, p=relevance_probability)[0]  # select next node
                next_node = node_neighbor[next]
                next_node_tab = node_neighbor_tab[next]
                paths[n].append(next_node_tab)
                if next_node == previous_node:  # terminating condition
                    if previous_node != root:
                        samples.append(current_node+self.n_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    def au_sample(self, a_node, tree, all_score):
        samples = []
        paths = []
        n = 0
        root = self.n_node
        while n < config.n_au_sample_gen:
            current_node = root
            previous_node = -1
            paths.append([])
            paths[n].append(a_node+self.n_node)
            while True:
                node_neighbor = tree[current_node]
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                node_neighbor_tab = []
                for i in node_neighbor:
                    node_neighbor_tab.append(a_node+self.n_node if i == root else i)
                relevance_probability = all_score[a_node+self.n_node, node_neighbor_tab] if current_node == root else all_score[current_node, node_neighbor_tab]
                relevance_probability = utils.softmax(relevance_probability)
                next = np.random.choice(range(len(node_neighbor)), size=1, p=relevance_probability)[0]  # select next node
                next_node = node_neighbor[next]
                next_node_tab = node_neighbor_tab[next]
                paths[n].append(next_node_tab)
                if next_node == previous_node:  # terminating condition
                    if previous_node != root:
                        samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths


    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """
        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs


    # @staticmethod
    def evaluation_save(self,epoch,d_epoch,g_epoch,flag=0):
        modes = [self.generator, self.discriminator]

        if config.app == "basic_evaluation":
            if flag == 0:
                print("------" + config.app + "* iteration: " + str(epoch))
                print("------" + config.app + "* iteration: " + str(epoch), file=self.fout)
                first_sim_s = []
                second_sim_s = []
                att_sim_s = []
                for i in range(2):
                    embedding_matrix = self.sess.run(modes[i].embedding_matrix)[0:self.n_node]
                    similarity = np.dot(embedding_matrix, embedding_matrix.T)
                    first_sim = first_order_proximity(similarity, config.fir_sim_label_filename)
                    first_sim_s.append(config.modes[i] + ": first-order proximity - AUC - " + str(first_sim))
                    print(first_sim_s[i])
                    print(first_sim_s[i], file=self.fout)
                    if os.path.isfile(config.sed_sim_label_filename):
                        second_sim = second_order_proximity(similarity, config.sed_sim_label_filename)
                        second_sim_s.append(config.modes[i] + ": second-order proximity - AUC - " + str(second_sim))
                        print(second_sim_s[i])
                        print(second_sim_s[i], file=self.fout)
                    if os.path.isfile(config.att_sim_label_filename):
                        att_sim = attribute_proximity(similarity, config.att_sim_label_filename)
                        att_sim_s.append(config.modes[i] + ": attribute proximity - AUC - " + str(att_sim))
                        print(att_sim_s[i])
                        print(att_sim_s[i], file=self.fout)
                    sio.savemat(config.emb_filenames[i] + str(epoch) + "_embedding.mat", {'embedding': embedding_matrix})
                    att_embedding_matrix = self.sess.run(modes[i].embedding_matrix)[self.n_node:(self.n_node+self.n_att)]
                    sio.savemat(config.emb_filenames[i] + str(epoch) + "_att_embedding.mat", {'embedding': att_embedding_matrix})
            elif flag == 1:
                print("--" + config.app  + str(epoch) + "* g_epoch: " + str(g_epoch))
                print("--" + config.app + "* iteration:" + str(epoch) + "-g_epoch" + str(g_epoch), file=self.fout)
                embedding_matrix = self.sess.run(self.generator.embedding_matrix)[0:self.n_node]
                similarity = np.dot(embedding_matrix, embedding_matrix.T)
                first_sim = first_order_proximity(similarity, config.fir_sim_label_filename)
                first_sim_s = "gen : first-order proximity - AUC - " + str(first_sim)
                print(first_sim_s)
                print(first_sim_s, file=self.fout)
                if os.path.isfile(config.sed_sim_label_filename):
                    second_sim = second_order_proximity(similarity, config.sed_sim_label_filename)
                    second_sim_s = "gen : second-order proximity - AUC - " + str(second_sim)
                    print(second_sim_s)
                    print(second_sim_s, file=self.fout)
                if os.path.isfile(config.att_sim_label_filename):
                    att_sim = attribute_proximity(similarity, config.att_sim_label_filename)
                    att_sim_s = "gen : attribute proximity - AUC - " + str(att_sim)
                    print(att_sim_s)
                    print(att_sim_s, file=self.fout)

                sio.savemat(config.emb_filenames[0] + str(epoch) + "-g_epoch" + str(g_epoch) + "_embedding.mat", {'embedding': embedding_matrix})
            else:
                print("--" + config.app + str(epoch) + "* d_epoch: " + str(d_epoch))
                print("--" + config.app + "* iteration:" + str(epoch) + "-d_epoch" + str(d_epoch), file=self.fout)
                embedding_matrix = self.sess.run(self.discriminator.embedding_matrix)[0:self.n_node]
                similarity = np.dot(embedding_matrix, embedding_matrix.T)
                first_sim = first_order_proximity(similarity, config.fir_sim_label_filename)
                first_sim_s = "dis : first-order proximity - AUC - " + str(first_sim)
                print(first_sim_s)
                print(first_sim_s, file=self.fout)
                if os.path.isfile(config.sed_sim_label_filename):
                    second_sim = second_order_proximity(similarity, config.sed_sim_label_filename)
                    second_sim_s = "dis : second-order proximity - AUC - " + str(second_sim)
                    print(second_sim_s)
                    print(second_sim_s, file=self.fout)
                if os.path.isfile(config.att_sim_label_filename):
                    att_sim = attribute_proximity(similarity, config.att_sim_label_filename)
                    att_sim_s = "dis : attribute proximity - AUC - " + str(att_sim)
                    print(att_sim_s)
                    print(att_sim_s, file=self.fout)

                sio.savemat(config.emb_filenames[1] + str(epoch) + "-d_epoch" + str(d_epoch) + "_embedding.mat", {'embedding': embedding_matrix})



    def train(self):
        # restore the model from the latest checkpoint if exists
        checkpoint = tf.train.get_checkpoint_state(config.model_log)
        if checkpoint and checkpoint.model_checkpoint_path and config.load_model:
            print("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        self.evaluation_save(0,0,0)
        print("-----------------------------------")
        print("-----------------------------------", file=self.fout)

        print("start training...")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)

            # save the model
            if epoch > 0 and epoch % config.save_steps == 0:
                self.saver.save(self.sess, config.model_log + str(epoch) + "model.checkpoint")

            center_nodes, neighbor_nodes, labels, node_1, node_2, reward = self.prepare_data_for_all()
            # D-steps
            for d_epoch in range(config.n_epochs_dis):
                print("d_epoch %d" % d_epoch)
                # training
                train_size = len(center_nodes)
                start_list = list(range(0, train_size, config.batch_size_dis))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + config.batch_size_dis
                    self.sess.run(self.discriminator.d_updates,
                                  feed_dict={self.discriminator.node_id: np.array(center_nodes[start:end]),
                                             self.discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                             self.discriminator.label: np.array(labels[start:end])})
                # if d_epoch % config.display == 0:
                #     print("** d_epoch %d" % d_epoch, file=self.fout)
                #     self.evaluation_save(epoch, d_epoch, 0, flag=2)

            # G-steps
            for g_epoch in range(config.n_epochs_gen):
                # print("g_epoch %d" % g_epoch)
                # training
                train_size = len(node_1)
                start_list = list(range(0, train_size, config.batch_size_gen))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + config.batch_size_gen
                    self.sess.run(self.generator.g_updates,
                                  feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                                             self.generator.node_neighbor_id: np.array(node_2[start:end]),
                                             self.generator.reward: np.array(reward[start:end])})
                # if g_epoch % config.display == 0:
                #     print("** g_epoch %d" % g_epoch, file=self.fout)
                #     self.evaluation_save(epoch, (config.n_epochs_dis-1), g_epoch, flag=1)

            if config.if_teacher_forcing:
                # prepare data for teacher forcing
                node_1 = []
                node_2 = []
                for i in self.root_nodes:
                    pos = self.uu_graph[i]
                    node_1.extend([i] * len(pos))
                    node_2.extend(pos)
                    pos = self.ua_graph[i]
                    pos = [j + self.n_node for j in pos]
                    node_1.extend([i] * len(pos))
                    node_2.extend(pos)
                for i in self.root_attnodes:
                    pos = self.au_graph[i]
                    node_1.extend([i + self.n_node] * len(pos))
                    node_2.extend(pos)
                reward = np.array([np.log(1 + np.exp(10))]*len(node_1))
                # training
                for g_epoch in range(config.n_epochs_gen_teacher):
                    # print("g_teacher_forcing_epoch %d" % g_epoch)
                    train_size = len(node_1)
                    start_list = list(range(0, train_size, config.batch_size_gen))
                    np.random.shuffle(start_list)
                    for start in start_list:
                        end = start + config.batch_size_gen
                        self.sess.run(self.generator.g_updates,
                                      feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                                                 self.generator.node_neighbor_id: np.array(node_2[start:end]),
                                                 self.generator.reward: np.array(reward[start:end])})

            self.evaluation_save(epoch, (config.n_epochs_dis-1),(config.n_epochs_gen-1))
            print("-----------------------------------")
            print("-----------------------------------", file=self.fout)
        print("training completes")
        self.fout.close()




if __name__ == "__main__": 
    model = ANGAN()
    model.train()
