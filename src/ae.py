import data_init as d_init
import experiments as exper
import plotter as plt
import lstm

import numpy as np
import numpy.random as rand

#hyperparameters
hp = {
    "ensemb_sz":      4,
    "bat_sz":         32,
    "epochs":         90,
    "feat_idxs":     [21],
    "subseq_len":     50,
    "subseq_stride":  1 }
hp[ "feat_dim" ] =    len(hp["feat_idxs"])
hp[ "label_dim"] =    len(hp["feat_idxs"]) 
hp[ "layer_dims" ] = [hp["subseq_len"], 50, hp["label_dim"]] 
hp[ "tr_set_prop" ] = 0.25

#
# top level
#
def main():
    
    cl_args   = d_init.read_cl_args()   
    sess_data = d_init.init_ae_dat(hp, cl_args)  

    #sess_data['clust_idxs'] = exper.km_clust(hp["ensemb_sz"], sess_data["subseqs"])
    gmm, clust_idxs         = exper.gmm_clust(hp["ensemb_sz"], sess_data["tr_set"]["feat_subseqs"])
    sess_data['clust_idxs'] = clust_idxs   

    #plt.plot_3d_subseqs(sess_data["subseqs"])
    #plt.plot_3d_clusts(sess_data["tr_set"]["feat_subseqs"], sess_data["clust_idxs"])

    sess_data["bat_idxs"] = d_init.bat_clust_subseqs(hp, sess_data)

    sess, lstm_graph, ref_params, ensemb_params = exper.train_from_clusts(hp, sess_data) 

    ensemb_err = lstm.test_ensemble(hp, sess_data, sess, lstm_graph, ensemb_params, gmm)
    lstm_err   = lstm.test_lstm(hp, sess_data, sess, lstm_graph, ref_params)    

    print("\n**ensemb MAE: " + str(np.mean(ensemb_err)))
    print("**src MAE: " + str(np.mean(lstm_err)))
    
 

if __name__ == '__main__':
    main()


