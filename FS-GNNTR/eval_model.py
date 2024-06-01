import torch
from gnntr_eval import GNNTR_eval
import statistics

def save_ckp(state, checkpoint_dir, filename):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)

def save_result(epoch, N, scores, filename):
            
    file = open(filename, "a")
    
    if epoch < N:
        file.write("Results: " + "\t")
        file.write(str(scores) + "\t")
    if epoch == N:
        file.write("Results: " + "\t")
        file.write(str(scores) + "\n")
        for i in scores:
            file.write("Support Sets: (Mean:"+ str(statistics.mean(i)) +", SD:" +str(statistics.stdev(i)) + ") | \t")
    file.write("\n")
    file.close()

dataset = "sider"
gnn = "gin"
support_set = 10
pretrained = "pre-trained/supervised_contextpred.pth"
baseline = 1
device = "cuda:0"

# FS-GNNTR - Two module GNN-TR architecture
# GraphSAGE - assumes that nodes that reside in the same neighborhood should have similar embeddings.
# GIN - Graph Isomorphism Network
# GCN - Standard Graph Convolutional Network

device = "cuda:0"      
model_eval = GNNTR_eval(dataset, gnn, support_set, pretrained, baseline)

print("Dataset:", dataset)

roc_auc_list = []
   
if dataset== "tox21":

    roc = [[],[],[]]
    f1s = [[],[],[]]
    prs = [[],[],[]]
    sns = [[],[],[]]
    sps = [[],[],[]]
    acc = [[],[],[]]
    bacc = [[],[],[]]

    labels =  ['SR-HSE', 'SR-MMP', 'SR-p53']

elif dataset == "sider":

    roc = [[],[],[],[],[],[]]
    f1s = [[],[],[],[],[],[]]
    prs = [[],[],[],[],[],[]]
    sns = [[],[],[],[],[],[]]
    sps = [[],[],[],[],[],[]]
    acc = [[],[],[],[],[],[]]
    bacc = [[],[],[],[],[],[]]

    labels =  ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']    
      
N = 30
   
for epoch in range(1, 31):
    
    [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], gnn_model, transformer_model, gnn_opt, t_opt = model_eval.meta_evaluate() #FS-GNNTR
        
    if epoch <= N:
        for i in range(len(roc_scores)):
            roc[i].append(round(roc_scores[i],4))
            f1s[i].append(round(f1_scores[i],4))
            prs[i].append(round(p_scores[i],4))
            sns[i].append(round(sn_scores[i],4))
            sps[i].append(round(sp_scores[i],4))
            acc[i].append(round(acc_scores[i],4))
            bacc[i].append(round(bacc_scores[i],4))
    
        save_result(epoch, N, roc, "results-exp/roc-GIN_10-sider.txt")
        save_result(epoch, N, f1s, "results-exp/f1s-GIN_10-sider.txt")
        save_result(epoch, N, prs, "results-exp/prs-GIN_10-sider.txt")
        save_result(epoch, N, sns, "results-exp/sns-GIN_10-sider.txt")
        save_result(epoch, N, sps, "results-exp/sps-GIN_10-sider.txt")
        save_result(epoch, N, acc, "results-exp/acc-GIN_10-sider.txt")
        save_result(epoch, N, bacc, "results-exp/bacc-GIN_10-sider.txt")
       
    
        #save_result(epoch, N, exp, "results-exp/mean-FS-GNNTR_tox21_10_new.txt")
    
