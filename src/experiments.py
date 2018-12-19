import sys
import tensorflow as tf
import numpy as np
from sklearn import cluster
from sklearn import mixture

import data_init as d_init
import lstm
import plotter as plt


#
#
#
def train_from_clusts(hp, sess_data):
    subseqs     = sess_data["feat_subseqs"]
    clust_idxs  = sess_data["clust_idxs"]

    lstm_graph = lstm.asmb_lstm(hp) 
    init_op    = tf.global_variables_initializer() 
    sess       = tf.Session()

    with sess.as_default() as _sess: 
        ensemb_params = lstm.ensemb_from_src(lstm_graph, hp["ensemb_sz"])  
        lstm.train_ensemble(hp, sess_data, _sess, lstm_graph, ensemb_params)

        return sess, ensemb_params   
#
# gaussian mixture model
#
def gmm_clust(clust_sz, feat_vecs):
    gmm_funct  = mixture.GaussianMixture(n_components=clust_sz, init_params='kmeans')
    clust_idxs = gmm_funct.fit_predict(feat_vecs)
    
    return gmm_funct, clust_idxs

#
# kmeans
#
def km_clust(clust_sz, feat_vecs):
    km_funct   = cluster.KMeans(n_clusters = clust_sz, init='k-means++')
    clust_idxs = km_funct.fit_predict(feat_vecs)
    return clust_idxs


