import init_helper as ihelp
import dsp
import plotter as plt
import numpy as np
import sys

import csv
import gc

#
# read cl arguments and return paths 
#
def read_cl_args():
    assert(len(sys.argv) >= 3), "usage: <input path(s)> <plot path>"
    csv_paths = [sys.argv[arg_idx] for arg_idx in range(1, len(sys.argv) - 1)]     
    plot_path = sys.argv[len(sys.argv) - 1]

    cl_args = {"csv_paths":csv_paths, "plot_path":plot_path} 
    return cl_args

#
# read data from files and assemble for training
#
def init_train_dat(hp, sess_specs, cl_args):
    print("\ninitializing session data")

    #read raw samples from files into [subject, feature dim]
    samps = read_samples(hp, sess_specs, cl_args["csv_paths"])

    #create subsequences
    subseqs = create_subseqs(hp, sess_specs, samps)

    #NOTE samples is dictionary with ["mp"] and ["ts"] streams
    return {"subseq_start_idxs":subseqs, "samples":samps}

    #commented out to test lazy subsequence assembly
    '''
    #split subseq indices into batches 
    bat_idxs = batch_subseq_idxs(hp, sess_specs, subseqs)  

    #gather subsequences from symbolic batches
    sess_dat = gather_bat_subseqs(sess_specs, bat_idxs, subseqs)        
    return sess_dat
    '''
#
# read file from path into memory and transpose, so row-major access returns complete series for each feature
#
def get_csv_cols(path):        
    csv_file = open(path)

    #copy into memory
    file_rows = csv.reader(csv_file, delimiter=',')
    mem_rows  = []

    for row in file_rows:
        mem_rows.append(row)
    csv_file.close()

    mem_cols = list(zip(*mem_rows))
    return mem_cols

#
# organize samples read from files
#
def read_samples(hp, sess_specs, paths):
    raw_samps = None

    if (sess_specs["mat_prof"] == True):
        raw_samps = mp_samps_from_files(hp, paths) 
    else:
        raw_samps = samps_from_files(hp, paths)
    
    #plt.plot_group( [ [raw_samps["ts"], "ts"], [raw_samps["mp"], "mp"]], "test.html")
    #sys.exit(0)

    return raw_samps
 
#
# returns { 'ts': time_series, 'mp': matrix_profile }
#
def mp_samps_from_files(hp, paths): 
    #NOTE using fixed file ordering for POC (timeseries_file, mp_pearcorr_file)
    assert(len(paths) == 2), "input files must be <time_series.csv> <mp.csv>"

    tr_set_sz    = hp["bat_sz"] * hp["bat_count"] 
    tst_set_sz   = int(tr_set_sz / hp["train_set_ratio"] * (1.0 - hp["train_set_ratio"])) + 1
    subseq_count = tr_set_sz + tst_set_sz      

    samp_streams = {}

    for path_idx in range(len(paths)):
        file_cols = ihelp.get_csv_cols(paths[path_idx])    

        if (path_idx == 0):
            samp_streams["ts"] = [float(samp) for samp in file_cols[0]]#[:201]#[:len(file_cols[0] - len(file_cols[0]) % 100]]#[:hp["seq_len"] - 1 + subseq_count]]
        else:
            samp_streams["mp"] = [float(samp) for samp in file_cols[0]]#[:201]#[:len(fil_cols[0] - len(file_cols[0]) % 100]]#[:subseq_count]]          

    #plt.plot_group( [ [samp_streams["ts"], "ts"], [samp_streams["mp"], "mp"]], "test.html")
    #sys.exit(0)

    return samp_streams

#
# returns [file_count, feature_dim, samp_count]
#
def samps_from_files(hp, paths): 
    feat_idxs   = hp["feat_idxs"]    
    bat_count   = hp["bat_count"]
    bat_sz      = hp["bat_sz"]
    seq_len     = hp["seq_len"]
    target_len  = hp["target_len"]     
    wind_stride = hp["wind_stride"]

    all_samp_streams = []

    for path in paths:
        file_cols         = ihelp.get_csv_cols(path)    
        feat_cols         = [file_cols[feat_idx] for feat_idx in feat_idxs]
        file_samp_streams = [[] for feat in feat_idxs]
    
        #subtract header from sample count
        samp_count = len(feat_cols[0]) - 1
    
        #prevent juxtaposition of samples from different files in subseq windows
        stride_count    = int((samp_count - (seq_len + target_len)) / wind_stride) 
        trim_samp_count = (stride_count * wind_stride) + (seq_len + target_len)          
        file_samps      = [[float(samp) for samp in feat_col[1:trim_samp_count + 1]] for feat_col in feat_cols] 

        all_samp_streams.append(file_samps)  
    return all_samp_streams

#
# normalize samples for all files
#
def normalize_samps(sess_specs, raw_samps): 
    normalizer = sess_specs["normalizer"]    

    if (normalizer != None):
        normalizer.normalize(raw_samps)

#
# partition samples in subsequences and follow any other specified steps before batch assembly 
#
def create_subseqs(hp, sess_specs, samps):
    subseqs = None
    if (sess_specs["mat_prof"] == True):
        subseqs = mp_subseqs_from_samps(hp, sess_specs, samps) 
    else:
        subseqs = subseqs_from_samps(hp, samps)
 
    if (sess_specs["plot_subseqs"] == True):
        ihelp.plot_subseqs(hp, subseqs)
    return subseqs

#
# similar to subseqs_from_samps, though labels are not taken from next immediate values, rather, mp values 
#
def mp_subseqs_from_samps(hp, sess_specs, samps):
    seq_len     = hp["seq_len"]
    target_len  = hp["target_len"]
    feat_dim    = hp["feat_dim"]
    label_dim   = hp["label_dim"]
    wind_stride = hp["wind_stride"]

    ts_samps = samps["ts"]
    mp_samps = samps["mp"]

    print("\n\n**# of TS samps: " + str(len(ts_samps)))
    print("\n**# of MP samps: " + str(len(mp_samps)))

    '''
    #FIXME normalize mp
    _mp_samps = [[samp] for samp in samps["mp"]]
    normalize_samps(sess_specs, _mp_samps)
    mp_samps = [samp[0] for samp in _mp_samps]
    '''

    tr_set_sz    = hp["bat_sz"] * hp["bat_count"] 
    tst_set_sz   = int(tr_set_sz / hp["train_set_ratio"] * (1.0 - hp["train_set_ratio"])) + 1
    subseq_count = tr_set_sz + tst_set_sz      

    #randomly draw subsequences
    print("\n**randomly drawing training subsequences from input time series")
    avail_start_idxs = np.arange(len(ts_samps) - seq_len + 1).tolist()
    rand_start_idxs  = [] 

    #draw training set subsequences
    for seq_idx in range(tr_set_sz):
        subseq_start_idx = np.random.randint(low=0, high=len(avail_start_idxs)-1)        
        rand_start_idxs.append(avail_start_idxs[subseq_start_idx])

    #select a contiguous range of the time series to select MP values in order
    #mp_tst_idx_start = avail_start_idxs[np.random.randint(low=0, high=len(avail_start_idxs) - 1 - tst_set_sz)] 
    rand_start_idxs += [tst_seq_start_idx for tst_seq_start_idx in range(int(len(mp_samps) / 2))]    

    feat_subseqs  = []
    label_subseqs = []

    print("\n\n**now assembling " + str(len(rand_start_idxs)) + " subsequences")
   
    #NOTE: commented out because subsequences will be extracted during testing 
    ''' 
    seq_count = len(rand_start_idxs)

    for seq_idx in range(seq_count):
        if (seq_idx % 100000 == 0):
            print("assembling subseq " + str(seq_idx))

        subseq_start_idx = rand_start_idxs[seq_idx]

        feat_wind  = [[feat] for feat in ts_samps[subseq_start_idx:subseq_start_idx + seq_len]]    
        label_wind = [[mp_samps[subseq_start_idx]]]

        if (seq_idx < tr_set_sz): 
            normalize_samps(sess_specs, feat_wind)
       
        feat_subseqs.append(feat_wind)
        label_subseqs.append(label_wind)  
    '''

    print("**subseqs extracted")

    '''    
    all_dat = {"feat_subseqs":feat_subseqs, "label_subseqs":label_subseqs} 
    return all_dat  
    ''' 

    return rand_start_idxs 

#
# assemble [subseq_count, subseq_len, subseq_dim] subseqs from [file_count, feat_dim, samp_cont] input 
#
def subseqs_from_samps(hp, samps):
    seq_len     = hp["seq_len"]
    target_len  = hp["target_len"]
    feat_dim    = hp["feat_dim"]
    label_dim   = hp["label_dim"]
    wind_stride = hp["wind_stride"]

    feat_subseqs  = []
    label_subseqs = []
 
    for file_samps in samps:
        #slice feature sample streams to create feature vectors
        feat_vecs    = [[feat_stream[samp_idx] for feat_stream in file_samps] for samp_idx in range(len(file_samps[0]))] 
        subseq_count = 1 + int((len(feat_vecs) - (seq_len + target_len)) / wind_stride) 
          
        for subseq_idx in range(subseq_count):
            start_idx = subseq_idx * wind_stride
            end_idx   = start_idx + seq_len

            feat_wind  = [feat_vec for feat_vec in feat_vecs[start_idx:end_idx]]    
            label_wind = [label_vec for label_vec in feat_vecs[end_idx:end_idx + target_len]]
  
            feat_subseqs.append(feat_wind)
            label_subseqs.append(label_wind)  
    return {"feat_subseqs":feat_subseqs, "label_subseqs":label_subseqs}

#
# symbolically form batches (wrapper)
#
def batch_subseq_idxs(hp, sess_specs, subseqs): 
    bat_idxs    = None
    clust_funct = sess_specs["clust_funct"]


    if (clust_funct != None):
        print("\nclustering subseqs") 

        #cluster subseqs in range up to delimiter; test sets will gather subseqs from the range starting thereafter
        set_delim_idx = hp["bat_count"] * hp["bat_sz"]  

        if (clust_funct.return_batches == True):
            #asmb_bats_sym will produce usable test set, so just reassign train set with that from clustering
            bat_idxs                = asmb_bats_symb(hp, sess_specs, subseqs)
            bat_idxs["tr_bat_idxs"] = clust_funct.clust(hp, subseqs["feat_subseqs"][:set_delim_idx])

        #batches will be formed by drawing evenly across clusters (must be uniformly sized)
        else:
            #clustering only operate on range up to train set size
            cl_idxs = clust_funct.clust(hp, subseqs["feat_subseqs"][:set_delim_idx])      
           
            if (sess_specs["plot_clusts"] == True):
                ihelp.plot_clusts(cl_idxs, subseqs["feat_subseqs"]) 
            bat_idxs = asmb_bats_symb(hp, sess_specs, subseqs, cl_idxs)              
    else:
        bat_idxs = asmb_bats_symb(hp, sess_specs, subseqs)   
    return bat_idxs
   
#
# batch subseq indices 
#
def asmb_bats_symb(hp, sess_specs, subseqs, clusts=None):    
    tr_bat_count    = hp["bat_count"]
    bat_sz          = hp["bat_sz"]
    train_set_ratio = hp["train_set_ratio"]
    feat_subseqs    = subseqs["feat_subseqs"]
    label_subseqs   = subseqs["label_subseqs"]

    tst_bat_count    = int((1.0 - train_set_ratio) * (tr_bat_count / train_set_ratio))      
    req_subseq_count = ((tr_bat_count + tst_bat_count) * bat_sz) + 1  

    assert(len(feat_subseqs) == len(label_subseqs)), "feat subseq count not equal to label subseq count"
    assert(req_subseq_count <= len(feat_subseqs)) \
          ,"not enough subsequences available for batch/set dimensions \n\trequired: " + str(req_subseq_count) + \
           "\n\tavailable: " + str(len(feat_subseqs)) 

    print("\nsmybollically creating batches")
    
    train_bats = []
    test_bats  = []   

    #collect training batch indices
    if (clusts == None):
        for bat_idx in range(tr_bat_count):
            start_idx = bat_idx * bat_sz
            train_bats.append([seq_idx for seq_idx in range(start_idx, start_idx + bat_sz)])                      
    else:  
        for bat_idx in range(tr_bat_count):
            train_bats.append([clusts[seq_idx][bat_idx] for seq_idx in range(bat_sz)]) 

    #collect test batch indices
    for bat_idx in range(tr_bat_count, tr_bat_count + tst_bat_count):
        start_idx = bat_idx * bat_sz       
        test_bats.append([seq_idx for seq_idx in range(start_idx, start_idx + bat_sz)])          
 
    bats = {"tr_bat_idxs":train_bats, "tst_bat_idxs":test_bats}
    return bats 

#
# gather mapped subseqs from batched indices
#
def gather_bat_subseqs(sess_specs, bat_idxs, subseqs):
    print("\ngathering concrete batch subsequences from symbolic")

    test_set = bats_from_idxs(bat_idxs["tst_bat_idxs"], subseqs)
    sess_dat = {"tr_set":None, "tst_set": test_set}

    if (sess_specs["kf_valid"] == True): 
        sess_dat["tr_set"] = kfold_bats_from_idxs(bat_idxs["tr_bat_idxs"], subseqs, sess_specs["fold_count"])    
    else:
        sess_dat["tr_set"] = bats_from_idxs(bat_idxs["tr_bat_idxs"], subseqs)
    return sess_dat

#
# returns dictionary of feature and label batches from subseq idxs
#
def bats_from_idxs(bat_idxs, subseqs):
    feat_subseqs  = subseqs["feat_subseqs"]
    label_subseqs = subseqs["label_subseqs"]

    feat_bats  = []
    label_bats = []

    for bat in bat_idxs:
        feat_bats.append([feat_subseqs[subseq_idx] for subseq_idx in bat])
        label_bats.append([label_subseqs[subseq_idx] for subseq_idx in bat])

    return {"feat_bats": feat_bats, "label_bats": label_bats}
   
#
# returns list of fold-specific dicts, containing training and validation sets: [{{tr_feats, tr_labels}, {valid_feats, valid_labels}}, fold2, fold3, ...]
#
def kfold_bats_from_idxs(bat_idxs, subseqs, fold_count):
    feat_subseqs  = subseqs["feat_subseqs"]
    label_subseqs = subseqs["label_subseqs"]

    assert(len(bat_idxs) % fold_count == 0), "fold count for kfold must evenly divide training set batches"           

    valid_bats_per_fold  = int(len(bat_idxs) / fold_count)
    folded_sess_dat = []

    for fold_idx in range(fold_count): 
        #validation batches
        valid_feat_bats  = []
        valid_label_bats = []

        valid_bats_start_idx = fold_idx * valid_bats_per_fold

        for valid_bat_idx in range(valid_bats_start_idx, valid_bats_start_idx + valid_bats_per_fold):
            valid_feat_bats.append([feat_subseqs[seq_idx] for seq_idx in bat_idxs[valid_bats_start_idx]])
            valid_label_bats.append([label_subseqs[seq_idx] for seq_idx in bat_idxs[valid_bats_start_idx]])                                  
        valid_set = {"feat_bats":valid_feat_bats, "label_bats":valid_label_bats}

        #training batches
        tr_feat_bats  = []
        tr_label_bats = [] 

        for tr_bat_idx in range(valid_bats_start_idx):
            tr_feat_bats.append([feat_subseqs[seq_idx] for seq_idx in bat_idxs[tr_bat_idx]])    
            tr_label_bats.append([label_subseqs[seq_idx] for seq_idx in bat_idxs[tr_bat_idx]])    

        for tr_bat_idx in range(valid_bats_start_idx + valid_bats_per_fold, len(bat_idxs)):
            tr_feat_bats.append([feat_subseqs[seq_idx] for seq_idx in bat_idxs[tr_bat_idx]])    
            tr_label_bats.append([label_subseqs[seq_idx] for seq_idx in bat_idxs[tr_bat_idx]])    
        tr_set = {"feat_bats":tr_feat_bats, "label_bats":tr_label_bats}
 
        folded_sess_dat.append({"tr_set":tr_set, "valid_set":valid_set})  
    return folded_sess_dat 

