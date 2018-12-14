import numpy as np
import csv
import plotly.offline as pltly
import plotly.graph_objs as graph_obj

#
# format sequences and respective labels for plotting 
#
def asmb_plot_dat(hp, seqs, labels): 
    sess_dat = []

    for seq_idx in range(len(seqs)):
        sess_dat.append([seqs[seq_idx], labels[seq_idx]])
    return sess_dat

#
# assemble labels and test predictions for plotting
#
def asmb_test_results(hp, tr_label_bats, tst_label_bats, test_predicts): 
    feat_dim   = hp["feat_dim"]    
    target_len = hp["target_len"]

    predicts_flat   = [[] for feat_idx in range(feat_dim)]
    tr_flat_labels  = [[] for feat_idx in range(feat_dim)]
    tst_flat_labels = [[] for feat_idx in range(feat_dim)]
    
    #flatten predictictions and labels
    for batch_predicts in test_predicts:
        for seq_predicts in batch_predicts:
            for predict_step in seq_predicts:
                for feat_idx in range(feat_dim):
                    predicts_flat[feat_idx].append(predict_step[feat_idx])
                    
    for tr_label_bat in tr_label_bats:
        for seq_labels in tr_label_bat:
            for step_labels in seq_labels:
                for feat_idx in range(feat_dim):
                    tr_flat_labels[feat_idx].append(step_labels[feat_idx])

                    #pad test set labels so that start of values aligns with ends of train set labels
                    tst_flat_labels[feat_idx].append(0)

    for tst_label_bat in tst_label_bats:
        for seq_labels in tst_label_bat:
            for step_labels in seq_labels:
                for feat_idx in range(feat_dim):          
                    tst_flat_labels[feat_idx].append(step_labels[feat_idx])
                    tr_flat_labels[feat_idx].append(0)

    for feat_idx in range(feat_dim):    	
        predicts_flat[feat_idx]   = np.array(predicts_flat[feat_idx]).astype(float)
        tr_flat_labels[feat_idx]  = np.array(tr_flat_labels[feat_idx]).astype(float)
        tst_flat_labels[feat_idx] = np.array(tst_flat_labels[feat_idx]).astype(float)
    
    assert(len(predicts_flat[0]) == len(tr_flat_labels[0]) == len(tst_flat_labels[0])), \
           "plot trace data are not equal lengths" 

    #FIXME plot error
    '''
    print("\n**prediction count: " + str(len(predicts_flat)))
    print("\n**label count:\n\ttr -" + str(len(tr_flat_labels)) + "\n\ttst -" + str(len(tst_flat_labels)))

    _err = [np.absolute(predicts_flat[tr_subseq_idx] - tr_flat_labels[tr_subseq_idx]) for tr_subseq_idx in range(len(tr_flat_labels))]
    _err += [np.absolute(predicts_flat[tst_subseq_idx + len(tr_flat_labels)] - tst_flat_labels[tst_subseq_idx]) for tst_subseq_idx in range(len(tst_flat_labels))    ]
    '''

    trace_dat = []
    for feat_idx in range(feat_dim):
        trace_dat.append([predicts_flat[feat_idx], "predicts_dim"+str(feat_idx)])
        trace_dat.append([tr_flat_labels[feat_idx], "tr_labels_dim"+str(feat_idx)])
        trace_dat.append([tst_flat_labels[feat_idx], "tst_labels_dim"+str(feat_idx)])

    #trace_dat.append([_err, "error"])

    return trace_dat

#
# writes ground truth and predictions into different columns at specified path 
#
def write_predicts_and_labels(predicts, sess_dat, path):

    #flatten batch predictions
    #predicts_flat = [predict[0] for bat_predicts in predicts \
    #                            for seq_predicts in bat_predicts \
    #                            for predict in seq_predicts] 
    #predicts_flat = [     predicts ] #FIXME
    predicts_flat = predicts

    label_samps = sess_dat["samples"] ["mp"]
    err = [np.abs(predicts_flat[predict_idx] - label_samps[predict_idx]) for predict_idx in range(len(predicts_flat))]

    #write output
    predict_count = len(predicts_flat)
    output_file   = open(path, "w") 

    #header
    output_file.write("predictions, labels, absolute_error\n")
 
    for predict_idx in range(predict_count):
        output_file.write(str(predicts_flat[predict_idx]) + "," + str(label_samps[predict_idx]) + "," + str(err[predict_idx]) + "\n") 

    #plot_group( [[predicts_flat, "predicts"], [label_samps[:len(predicts_flat)], "labels"], [err, "error"]], "predictions.html")
  
#
# plot pairs of [trace_data, identifier] and write to supplied path 
#
def plot_group(data_id_pairs, output_path):
    traces = []

    for trace_dat in data_id_pairs:
        traces.append(graph_obj.Scatter(x    = np.arange(len(trace_dat[0])),
                                        y    = trace_dat[0],
                                        mode = 'lines',
                                        name = trace_dat[1]))
    pltly.plot(traces, filename = output_path)


