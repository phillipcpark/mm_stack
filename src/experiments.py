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
    lstm_graph = lstm.asmb_lstm(hp) 
    init_op    = tf.global_variables_initializer() 
    sess       = tf.Session()

    with sess.as_default() as _sess: 
        #initial training 
        init_params = lstm_graph["lstm_cells"].get_weights()
        hp["epochs"] = 20
        #lstm.train(hp, sess_data, _sess, lstm_graph, [init_params])

        #clone source model for ensemble
        ref_params, ensemb_params = lstm.ensemb_from_src(lstm_graph, hp["ensemb_sz"]) 

        #train ensemble 
        hp["epochs"] = 40
        lstm.train_ensemble(hp, sess_data, _sess, lstm_graph, ensemb_params)
        #train source
        hp["epochs"] = 10
        lstm.train(hp, sess_data, _sess, lstm_graph, ref_params)

        return sess, lstm_graph, ref_params, ensemb_params   
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


