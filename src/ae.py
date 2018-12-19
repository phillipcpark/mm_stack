import data_init as d_init
import experiments as exper
import plotter as plt

import numpy.random as rand

#hyperparameters
hp = {
    "ensemb_sz":      4,
    "bat_sz":         64,
    "epochs":         11,
    "feat_idxs":     [21],
    "subseq_len":     3,
    "subseq_stride":  15 }
hp[ "feat_dim" ] =    len(hp["feat_idxs"])
hp[ "label_dim"] =    len(hp["feat_idxs"]) 
hp[ "layer_dims" ] = [50, 50, hp["label_dim"]] 
hp[ "tr_set_prop" ] = 0.5

#
# top level
#
def main():
    cl_args   = d_init.read_cl_args()   
    sess_data = d_init.init_ae_dat(hp, cl_args)  

    #sess_data['clust_idxs'] = exper.km_clust(hp["ensemb_sz"], sess_data["subseqs"])
    gmm, clust_idxs = exper.gmm_clust(hp["ensemb_sz"], sess_data["feat_subseqs"])
    sess_data['clust_idxs'] = clust_idxs   


    #plt.plot_3d_subseqs(sess_data["subseqs"])
    #plt.plot_3d_clusts(sess_data["feat_subseqs"], sess_data["clust_idxs"])

    sess_data["bat_idxs"] = d_init.bat_clust_subseqs(hp, sess_data)
    sess, ensemb_params = exper.train_from_clusts(hp, sess_data) 

    print(str(ensemb_params[0][0]) + "\n\n" + str(ensemb_params[-1][0]))


if __name__ == '__main__':
    main()


