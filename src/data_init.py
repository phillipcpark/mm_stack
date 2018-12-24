import numpy as np
import sys
import csv

#
# read cl arguments and return paths 
#
def read_cl_args():
    assert(len(sys.argv) == 2), "usage: <input path>"
    csv_paths = [sys.argv[1]]     

    cl_args = {"csv_paths":csv_paths} 
    return cl_args

#
# read data from files and assemble for training
#
def init_ae_dat(hp, cl_args):
    headers, cols   = read_file(cl_args["csv_paths"][0])
    tr_set, tst_set = create_subseqs(hp, cols)
 
    return {"attr_headers": headers, "tr_set": tr_set, "tst_set": tst_set} 
   
#
# reads headers and columns from files into memory and returns references 
#
def read_file(path):
    csv_file  = open(path)      
    file_rows = csv.reader(csv_file, delimiter=',')    
    mem_rows  = []

    for row in file_rows:
        mem_rows.append(row)
    csv_file.close()

    headers  = mem_rows[0]
    mem_cols = list(zip(*mem_rows[1:]))

    return headers, mem_cols

#
# normalizes sequence by (x - min) / range 
#
def minmax_norm(seq):
    seq_min = np.amin(seq)
    seq_max = np.amax(seq)
    seq_range = seq_max - seq_min

    _seq = np.array([float(samp - seq_min) / seq_range for samp in seq]) 
    return _seq

#
# creates subsequences from columns of data
#
def create_subseqs(hp, cols):
    extract_idxs  = hp["feat_idxs"]    
    subseq_len    = hp["subseq_len" ]
    subseq_stride = hp["subseq_stride"]  
 
    total_seq_len = len(cols[0])  
    subseq_count  = None
   
    if (subseq_stride == 1): 
        subseq_count = total_seq_len - subseq_len - 1
    else:
        subseq_count = int(float(total_seq_len - subseq_len) / subseq_stride)

    #normalize time series
    for col_idx in range(1, len(cols)):
        cols[col_idx] = minmax_norm([float(val) for val in cols[col_idx]])

    subseqs = []
    labels  = [] 

    for subseq_idx in range(subseq_count):
        _subseq = []

        for samp_idx in range(subseq_len):
            _samp = float(cols[extract_idxs[0]][subseq_idx * subseq_stride + samp_idx])
            _subseq.append(_samp)
 
        subseqs.append(_subseq) 
        labels.append(float(cols[extract_idxs[0]][subseq_idx * subseq_stride + subseq_len]))

    #shuffle
    shuff_idxs = np.arange(subseq_count)
    np.random.shuffle(shuff_idxs)

    _subseqs = [subseqs[_idx] for _idx in shuff_idxs]
    _labels  = [labels[_idx] for _idx in shuff_idxs]

    #number of subsequences for training set
    tr_count = int(hp["tr_set_prop"] * subseq_count)

    tr_set  = {"feat_subseqs": _subseqs[:tr_count], "labels": _labels[:tr_count]}
    tst_set = {"feat_subseqs": _subseqs[tr_count:], "labels": _labels[tr_count:]}

    #FIXME
    print("\n**tr set sz: " + str(len(tr_set["feat_subseqs"])))
    print("\n**tst set sz: " + str(len(tst_set["feat_subseqs"])))
 
 
    return tr_set, tst_set

#
# maps subsequences to batches
#
def bat_clust_subseqs(hp, sess_data):
    clust_count = hp["ensemb_sz"]
    clust_idxs  = sess_data["clust_idxs"]
    bat_sz      = hp["bat_sz"]

    #gather subseq idxs by cluster 
    clust_subseq_idxs = [ [] for clust in range(clust_count) ]

    for subseq_idx in range(len(clust_idxs)):
        clust_idx = clust_idxs[subseq_idx]
        clust_subseq_idxs[clust_idx].append(subseq_idx)    

    #split each list of cluster subseqs into batches
    clust_bat_idxs = []
    
    for clust_idx in range(clust_count):
        clust_bat_idxs.append(bat_subseq_idxs(clust_subseq_idxs[clust_idx], bat_sz)) 
       
    return clust_bat_idxs

#
#
#
def bat_subseq_idxs(subseq_idxs, bat_sz): 
    seq_count = len(subseq_idxs)
    bat_count = int(seq_count / bat_sz)
    bat_idxs  = [subseq_idxs[bat_idx * bat_sz:(bat_idx * bat_sz) + bat_sz] for bat_idx in range(bat_count)]
    
    return bat_idxs











