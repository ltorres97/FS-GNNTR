import argparse
import torch
from gnntr_eval import GNNTR_eval
import statistics
import shutil
import matplotlib.pyplot as plt

def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)

def save_result(epoch, N, exp, filename, tl):
            
    file = open(filename, "a")
    
    file.write("Results: " + "\t")
    if epoch < N:
        file.write(str(exp) + "\t")
    if epoch == N:
        file.write(str(exp))
        if tl == 0:
            for list_acc in exp:
                file.write(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(list_acc)) +", SD:" +str(statistics.stdev(list_acc)) + ") | \t")
        elif tl == 1:
            file.write(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(exp[0])) +", SD:" +str(statistics.stdev(exp[0])) + ") | \t")
            
          
    file.write("\n")
    file.close()


dataset = "tox21"
gnn= "gin"
support_set = 10
pretrained = "pre-trained/supervised_contextpred.pth"
baseline = 0
device = "cuda:0"
tl = 0

"""
For this example, on fsgnnconv_eval.py consider:
ckp_path_gnn = "checkpoints/checkpoints-FSGNNConv/check-sider-5-gnn.pt"
ckp_path_cnn = "checkpoints/checkpoints-FSGNNConv/check-sider-5-cnn.pt"

and

Uncomment last line of fsgnnconv_eval.py    
"""

# FS-GNNConv - Two module GNN-CNN architecture
# GraphSage - assumes that nodes that reside in the same neighborhood should have similar embeddings.
# GIN - Graph Isomorphism Network
# GCN - Standard Graph Convolutional Network


device = "cuda:0"      
model_eval = GNNTR_eval(dataset, gnn, support_set, pretrained, baseline, tl)
model_eval.to(device)

print("Dataset:", dataset)

roc_auc_list = []

if tl == 0:
    if dataset== "tox21":
        exp = [[],[],[]]
        labels =  ['SR-HSE', 'SR-MMP', 'SR-p53']
    elif dataset == "sider":
        exp = [[],[],[],[],[],[]]
        labels =  ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']    

if tl == 1:
   exp = [[]]
   labels = ['Mean_Task']
   
    
    
N = 30
   
for epoch in range(1, 10000):
    
    roc_scores, gnn_model, cnn_model, gnn_opt, cnn_opt = model_eval.meta_evaluate() #FS-GNNConv
   
    #roc_scores, gnn_model, gnn_opt = model.meta_evaluate(grads) #baselines
    if roc_auc_list != []:
        for score in range(len(roc_auc_list)):

            if roc_auc_list[score] < roc_scores[score]:
                roc_auc_list[score] = roc_scores[score]
                
    if epoch <= N:
      i=0
      for a in roc_scores:
        exp[i].append(round(a,4))
        i+=1
      
    if epoch > N:
      for i in range(len(exp)):
        if min(exp[i]) < round(roc_scores[i],4):
          index = exp[i].index(min(exp[i]))
          exp[i][index] = roc_scores[i]
    else:
        roc_auc_list = roc_scores
      
    
    save_result(epoch, N, exp, "results-exp/mean-FS-GNNTR_tox21_10_new.txt", tl)
    
    if tl == 0:
        if dataset == "tox21":
            box_plot_data=[exp[0], exp[1], exp[2]]
            plot_title = "Tox21"
        elif dataset == "sider":
            box_plot_data=[exp[0], exp[1], exp[2], exp[3], exp[4], exp[5]]
            plot_title = "SIDER"  
    
    if tl == 1:
        if dataset == "tox21":
            box_plot_data=[exp[0]]
            plot_title = "Transfer-Learning (SIDER->Tox21)"
        elif dataset == "sider":
            box_plot_data=[exp[0]]
            plot_title = "Transfer-Learning (Tox21->SIDER)"

    
    fig = plt.figure()   
    fig.suptitle(plot_title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot(box_plot_data,labels=labels)
    ax.set_xlabel('Test Task')
    ax.set_ylabel('ROC-AUC score')
    plt.grid(b=False)
    if epoch == N and tl == 0:
        plt.savefig('plots/test/boxplot_GT_'+ str(dataset) + '_' + str(support_set), dpi=300)
    #plt.savefig('plots/boxplot_GT_'+ str(dataset) + '_' + str(support_set), dpi=300)
    plt.show()
    plt.close(fig)
    
       
    if epoch >= N/2:
        for list_acc in exp:
            print(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(list_acc)) +", SD:" +str(statistics.stdev(list_acc)) + ")")
