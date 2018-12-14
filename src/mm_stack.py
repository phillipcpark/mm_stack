import dat_init as d_init
import dsp
import lstm
import plotter as plt
import seq_clust as seqcl

import numpy as np
import tensorflow as tf

#hyperparameters
hp = {    
    "epochs"            : 0,#301,
    "bat_sz"            : 256,      
    "bat_count"         : int(20000000/256),   
    "feat_idxs"         : [0], #specify columns to extract from input file
    "seq_len"           : 100, #feature subsequence len 
    "target_len"        : 1 }  #number of points to predict (to-do)
hp[ "feat_dim" ]        = len(hp["feat_idxs"])   
hp[ "label_dim" ]       = hp["feat_dim"]
hp[ "hidden_dims" ]     = [100, 100, hp["label_dim"]] #layer sizes; len of list specifies number of layers per LSTM
hp[ "wind_stride" ]     = 1 
hp[ "train_set_ratio" ] = 0.00000005 #ratio between train set sz (bat_sz * bat_count) and test set
hp[ "clip_thresh" ]     = 3.0 #for gradient clip 

#functor references and other experiment-specific details
sess_specs = {  
           "mat_prof"   : True, 
           "kf_valid"   : False, #session will do k-fold cross validation 
           "fold_count" : 6, 

           #specify functor to precluster training data (set to None for no preclustering) 
           "clust_funct" : None, #seqcl.seq_clust(cl_strat = seqcl.depth_bat_strat(), return_batches=True), 

           #specify how time series will be normalized (before any windowing or batching)
           #    all_subj: if set, normalize time series from all files together                                 
           #    all_dim: if set, normalize all feature dimensions together(within all_subj scope)
           "norm_axes_scope" : {"all_subj":False, "all_dim":False} }  
sess_specs["normalizer"   ]  = dsp.mean_normalizer()#axes_scope = sess_specs["norm_axes_scope"])

sess_specs["tr_out_interval"] = 1     #interval at which training/validation error is recorded     
sess_specs["shuff_subseqs"]   = True  #shuffle before batching subsequences
sess_specs["plot_subseqs" ]   = False
sess_specs["plot_clusts"  ]   = False

#
# top level logic
#
def main():
    #read training data from files and assemble lstm
    cl_args    = d_init.read_cl_args() 

    #NOTE: for lazy subsequence assembly testing, sess_dat is dictionary with "subseq_start_idxs" and "samples"
    sess_dat   = d_init.init_train_dat(hp, sess_specs, cl_args)
    lstm_graph = lstm.asmb_lstm(hp)

    #train lstm and plot error
    #sess, train_err = lstm.train_for_err(hp, sess_specs, lstm_graph, sess_dat)
    #print("\n\n**training model")
    #sess = lstm.train_for_err(hp, sess_specs, lstm_graph, sess_dat)

    init_op = tf.global_variables_initializer()                                                    
 
    #sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))                                                                                             
    sess.run(init_op)   
 
    print("\n\n**testing model on data set")
    predicts = lstm.test_predicts(hp, sess, lstm_graph, sess_dat, restore_path="../params_cp/seq4p2m_thresh0p75/cp-100")
    #plot_dat = plt.asmb_test_results(hp, sess_dat["tr_set"]["label_bats"], sess_dat["tst_set"]["label_bats"], predicts) 
   
    print("\n\n**writing predictions and labels")
    plt.write_predicts_and_labels(predicts, sess_dat, "train4p2m_predict20m.csv")
    
    #err_labels      = ["batch_err", "avg_ep_err", "valid_bat_err", "valid_avg_ep_err"]
    #plot_dat        = plt.asmb_plot_dat(hp, train_err, err_labels)

    #plt.plot_group(plot_dat, cl_args["plot_path"])
    
if __name__ == '__main__':
    main()


