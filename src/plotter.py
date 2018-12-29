import numpy as np
import plotly.offline as plt
import plotly.graph_objs as graph_obj
import sys

#
#
#
def plot_residuals(predicts, labels):
    traces = []

    for comp_idx in range(len(predicts[0])):
        predicts_trace = graph_obj.Scatter(x = np.arange(len(predicts)), \
                                           y = [predict[comp_idx] for predict in predicts],
                                           mode = 'lines',
                                           name = 'predict_'+str(comp_idx))
    
        labels_trace = graph_obj.Scatter(x = np.arange(len(predicts)), \
                                         y = [label[comp_idx] for label in labels],
                                      mode = 'lines',
                                     name = 'gt_'+str(comp_idx))
        traces.append(predicts_trace)
        traces.append(labels_trace)

    plt.plot(traces, filename='residuals.html') 

                                      




#
#
#
def plot_3d_clusts(subseqs, clust_idxs):

    '''
    #specify how subsequence will be summarized into 3 dimensions
    rep_idxs = [0, int(float(len(subseqs[0])) / 3), 2 * int(float(len(subseqs[0])) / 3)] 
    vecs_3d  = []

    for subseq in subseqs:
        redux_3d = []
        redux_3d.append(np.mean(subseq[rep_idxs[0]:rep_idxs[1]]))    
        redux_3d.append(np.mean(subseq[rep_idxs[1]:rep_idxs[2]]))
        redux_3d.append(np.mean(subseq[rep_idxs[2]:]))
            
        vecs_3d.append(redux_3d) 
    ''' 
   

    #gather coordinates by cluster 
    num_clusts = len(np.unique(clust_idxs)) 
    clust_subseqs = [ [] for clust in range(num_clusts) ]

    for subseq_idx in range(len(subseqs)): #range(len(vecs_3d)):
        clust_idx = clust_idxs[subseq_idx]
        #clust_subseqs[clust_idx].append(vecs_3d[subseq_idx]) 
 
        #NOTE flatten component dimensions      
        clust_subseqs[clust_idx].append([subseqs[subseq_idx][0][comp_idx] for comp_idx \
                                                                          in range(len(subseqs[subseq_idx][0]))] )


 
    traces = [] 
    for clust_idx in range(num_clusts):
        _trace = graph_obj.Scatter3d(
            x = np.array([subseq[0] for subseq in clust_subseqs[clust_idx]]),
            y = np.array([subseq[1] for subseq in clust_subseqs[clust_idx]]), 
            z = np.array([subseq[2] for subseq in clust_subseqs[clust_idx]]),
            name = "clust_"+str(clust_idx),
            mode = 'markers',
            marker = dict(
                size = 5,
                line = dict(
                    width = 2
                ) 
            )   
        )   
        traces.append(_trace) 
    plt.plot(traces, filename="clusts.html")   
    

#
# 
#
def plot_3d_subseqs(subseqs):

    #specify how subsequence will be summarized into 3 dimensions
    rep_idxs = [0, int(float(len(subseqs[0])) / 3), 2 * int(float(len(subseqs[0])) / 3)] 
    vecs_3d  = []

    for subseq in subseqs:
        redux_3d = []
        redux_3d.append(np.mean(subseq[rep_idxs[0]:rep_idxs[1]]))    
        redux_3d.append(np.mean(subseq[rep_idxs[1]:rep_idxs[2]]))
        redux_3d.append(np.mean(subseq[rep_idxs[2]:]))
            
        vecs_3d.append(redux_3d) 

    traces = []
    traces.append(graph_obj.Scatter3d(x    = np.array([vec[0] for vec in vecs_3d]),
                                      y    = np.array( [vec[1] for vec in vecs_3d]),
                                      z    = np.array( [vec[2] for vec in vecs_3d]),
                                      mode = 'markers',
                                      marker = dict(
                                          size = 2
                                      )))
    plt.plot(traces, filename = '3d_ts.html')

#
#
#
def plot_2d_subseqs(subseqs, tag='subseq_'):
    traces = []

    for subseq_idx in range(len(subseqs)):
        traces.append(graph_obj.Scatter(x    = np.arange(len(subseqs[subseq_idx])),
                                        y    = subseqs[subseq_idx],
                                        mode = 'lines',
                                        name = tag+"_"+str(subseq_idx)))

    plt.plot(traces, filename = 'subseqs.html')


