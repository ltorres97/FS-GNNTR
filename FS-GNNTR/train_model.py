import torch
from gnntr_train import GNNTR

def save_ckp(state, checkpoint_dir, filename):
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)

dataset = "tox21"
gnn= "gin" #gin, graphsage, gcn
support_set = 10
pretrained = "pre-trained/supervised_contextpred.pth"
baseline = 0
device = "cuda:0"
model = GNNTR(dataset, gnn, support_set, pretrained, baseline)
model.to(device)

if dataset == "tox21":
    exp = [0,0,0]
    labels =  ['SR-HSE', 'SR-MMP', 'SR-p53']
elif dataset == "sider":
    exp = [0,0,0,0,0,0]
    labels =  ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.P.P.C.']

    
for epoch in range(1, 10000):
    
    model.meta_train()
    
    roc_scores, gnn_model, transformer_model, gnn_opt, t_opt = model.meta_test() 
    
    print(roc_scores)
    checkpoint_gnn = {
            'epoch': epoch + 1,
            'state_dict': gnn_model,
            'optimizer': gnn_opt
    }
    
    checkpoint_transformer = {
            'epoch': epoch + 1,
            'state_dict': transformer_model,
            'optimizer': t_opt
    }

    checkpoint_dir = 'checkpoints/checkpoints-GT/'
    
    for i in range(0, len(roc_scores)):
        if exp[i] < roc_scores[i]:
            exp[i] = roc_scores[i]
            is_best = True
            
    filename = "results-exp/FS-GNNTR-tox21-10-new.txt"
    file = open(filename, "a")
    file.write("ROC-AUC scores:\t")
    #file.write(str(exp))
    file.write(str(roc_scores)+"\t ; ")
    file.write(str(exp)) 
    file.write("\n")
    file.close()
    
    if baseline == 0:
        save_ckp(checkpoint_gnn, checkpoint_dir, "/FS-GNNTR_GNN_tox21_10_2.pt")
        save_ckp(checkpoint_transformer, checkpoint_dir, "/FS-GNNTR_Transformer_tox21_10_2.pt")

    elif baseline == 1:
        save_ckp(checkpoint_gnn, checkpoint_dir, "/GT_GNN_sider_5.pt")
        
   