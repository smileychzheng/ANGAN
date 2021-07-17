import numpy as np
from sklearn.model_selection import train_test_split
from liblinearutil import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb
import scipy.io as sio

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def evaluation(train_vec, test_vec, train_y, test_y, classifierStr='SVM', normalize=0):

    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    else:
        # print('Training SVM classifier...')
        classifier = LinearSVC()

    if(normalize == 1):
        print('Normalize data')
        allvec = list(train_vec)
        allvec.extend(test_vec)
        allvec_normalized = preprocessing.normalize(allvec, norm='l2', axis=1)
        train_vec = allvec_normalized[0:len(train_y)]
        test_vec = allvec_normalized[len(train_y):]

    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    acc = accuracy_score(test_y, y_pred)

    macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')


    return acc, macro_f1, micro_f1

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def getSimilarity(result):
    return np.dot(result, result.T)
    
def check_link_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        # print ("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        n_node = len(data)
        for ind in sortedInd:
            x = ind // n_node
            y = ind % n_node
            count += 1
            if ((y in data[x]) or x == y):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        ret.append(precisionK[index - 1])
    return ret


def first_order_proximity(similarity, label_file):
    fin = sio.loadmat(label_file)
    label = fin['fir_sim_label']
    AUC = metrics.roc_auc_score(label.T, similarity.reshape(-1).T)

    return AUC

def second_order_proximity(similarity, label_file):
    fin = sio.loadmat(label_file)
    label = fin['sed_sim_label']
    AUC = metrics.roc_auc_score(label.T, similarity.reshape(-1).T)

    return AUC

def attribute_proximity(similarity, label_file):
    fin = sio.loadmat(label_file)
    label = fin['att_sim_label']
    AUC = metrics.roc_auc_score(label.T, similarity.reshape(-1).T)

    return AUC



