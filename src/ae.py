import sys
import data_init as d_init
import experiments as exper
import plotter as plt
import lstm

import numpy as np
import numpy.random as rand

import sklearn.metrics as metrics


#hyperparameters
hp = {
    "ensemb_sz":      3,
    "bat_sz":         32,
    "epochs":         90,
    "feat_idxs":      [1, 21], #[4, 5, 6, 8, 21],
    "subseq_len":     32, #1,
    "subseq_stride":  1 }
hp[ "feat_dim" ] =    len(hp["feat_idxs"])
hp[ "label_dim"] =    len(hp["feat_idxs"]) 
hp[ "layer_dims" ] = [hp["subseq_len"], 100, 100, hp["label_dim"]] 
hp[ "tr_set_prop" ] = 0.3 

#
# top level
#
def main():

    print("\n**initializing session data")    
    cl_args   = d_init.read_cl_args()   
    sess_data = d_init.init_ae_dat(hp, cl_args)  

    print("\n**clustering with gmm")
    gmm, clust_idxs         = exper.gmm_clust(hp["ensemb_sz"], sess_data["tr_set"]["feat_subseqs"])
    sess_data['clust_idxs'] = clust_idxs   

    #plt.plot_3d_subseqs(sess_data["subseqs"])
    #plt.plot_3d_clusts(sess_data["tr_set"]["feat_subseqs"], sess_data["clust_idxs"])

    sess_data["bat_idxs"] = d_init.bat_clust_subseqs(hp, sess_data)

    print("\n**training")
    sess, lstm_graph, ref_params, ensemb_params = exper.train_from_clusts(hp, sess_data) 

    print("\n**testing")
    ensemb_predicts, ensemb_err = lstm.test_ensemble(hp, sess_data, sess, lstm_graph, ensemb_params, gmm)
    #predicts, lstm_err   = lstm.test_lstm(hp, sess_data, sess, lstm_graph, ref_params)    
 
    print("\n**ensemb RMSE: " + str(np.sqrt(np.mean(ensemb_err))))
    #print("**src RMSE: " + str(np.sqrt(np.mean(lstm_err))))

    #print("\n\nsrc err distr, ensemb err distr:\n")
    #for err_idx in range(len(ensemb_err)):
    #    print(str(np.sqrt(lstm_err[err_idx])) + ", " + str(np.sqrt(ensemb_err[err_idx])))

    r2 = metrics.r2_score(sess_data["tst_set"]["labels"], ensemb_predicts)
    print("\n**r2: " + str(r2)) 

    plt.plot_residuals(ensemb_predicts, sess_data["tst_set"]["labels"])


if __name__ == '__main__':
    main()


