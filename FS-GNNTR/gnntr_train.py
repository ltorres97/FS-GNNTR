import torch
import torch.nn as nn
from transformer import GNN_prediction, TR
import torch.nn.functional as F
from data import MoleculeDataset, random_sampler
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from sklearn.manifold import TSNE
# from tsnecuda import TSNE # Use this package if the previous one doesn't work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']


def sample_train(n_tasks, data, batch_size, n_support, n_query):
    
    support_sets = []
    query_sets = []
    
    for train_task in range(n_tasks):
         dataset = MoleculeDataset("Data/" + data + "/pre-processed/task_" + str(train_task+1), dataset = data)
         support_dataset, query_dataset = random_sampler(dataset, data, train_task, n_support, n_query, train=True)
         support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
         query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
         support_sets.append(support_set)
         query_sets.append(query_set)
         
    return support_sets, query_sets

def sample_test(tasks, test_task, data, batch_size, n_support, n_query):
    
    dataset = MoleculeDataset("Data/" + data + "/pre-processed/task_" + str(tasks-test_task), dataset = data)
    support_dataset, query_dataset = random_sampler(dataset, data, tasks-test_task-1, n_support, n_query, train=False)
    support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
    
    return support_set, query_set
    

def roc_accuracy(roc_scores, y_label, y_pred):
    
    roc_auc_list = []
    y_label = torch.cat(y_label, dim = 0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().detach().numpy()
   
    roc_auc_list.append(roc_auc_score(y_label, y_pred))
    roc_auc = sum(roc_auc_list)/len(roc_auc_list)
    roc_scores.append(roc_auc)    
    
    return roc_scores

def parse_pred(logit):
    
    pred = F.sigmoid(logit)
    pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
    pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred) 
    
    return pred


def plot_tsne(nodes, labels, t):
    
    #Plot t-SNE visualizations
    
    labels_tox21 = ['SR-HSE', 'SR-MMP', 'SR-p53']
    #labels_sider = ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.S.P.C.']
     
    t+=1
    node_emb_tsne = np.asarray(nodes)
    y_tsne = np.asarray(labels).flatten()
    slipper_colour = pd.DataFrame({'colour': ['Blue', 'Orange'],
                       'label': [0, 1]})
    
    c_dict = {'Positive': '#ff7f0e','Negative': '#1f77b4' }
     
    z = TSNE(n_components=2, init='random').fit_transform(node_emb_tsne)
    label_vals = {0: 'Negative', 1: 'Positive'}
    tsne_result_df = pd.DataFrame({'tsne_dim_1': z[:,0], 'tsne_dim_2': z[:,1], 'label': y_tsne})
    tsne_result_df['label'] = tsne_result_df['label'].map(label_vals)
    fig, ax = plt.subplots(1)
    sns.set_style("ticks",{'axes.grid' : True})
    g1 = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', data=tsne_result_df, ax=ax,s=10, palette = c_dict, hue_order=('Negative', 'Positive'))
    lim = (z.min()-5, z.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal') 
    
    g1.legend(title=labels_tox21[t-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)    
    g1.set(xticklabels=[])
    g1.set(yticklabels=[])
    g1.set(xlabel=None)
    g1.set(ylabel=None)
    g1.tick_params(bottom=False) 
    g1.tick_params(left=False)
    plt.savefig('plots/'+labels_tox21[t-1])
    plt.show()
    plt.close(fig)
    
    return t
    
class GNNTR(nn.Module):
    def __init__(self, dataset, gnn, support_set, pretrained, baseline):
        super(GNNTR,self).__init__()
        
        if dataset== "tox21":
            self.tasks = 12
            self.train_tasks  = 9
            self.test_tasks = 3

        elif dataset == "sider":
            self.tasks = 27
            self.train_tasks  = 21
            self.test_tasks = 6
            
        self.data = dataset
        self.baseline = baseline
        self.graph_layers = 5
        self.n_support = support_set
        self.learning_rate = 0.001
        self.n_query = 128
        self.emb_size = 300
        self.batch_size = 10
        self.lr_update = 0.4
        self.k_train = 5
        self.k_test = 10
        self.device = 0
        self.loss = nn.BCEWithLogitsLoss()
        self.gnn = GNN_prediction(self.graph_layers, self.emb_size, jk = "last", dropout_prob = 0.5, pooling = "mean", gnn_type = gnn)
        self.transformer = TR(300, (30,1), 1, 128, 5, 5, 256) 
        self.gnn.from_pretrained(pretrained)
        self.pos_weight = torch.FloatTensor([25]).to(self.device) #Tox21: 35; SIDER: 1
        self.loss_transformer = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.meta_opt = torch.optim.Adam(self.transformer.parameters(), lr=1e-5)
        
        graph_params = []
        graph_params.append({"params": self.gnn.gnn.parameters()})
        graph_params.append({"params": self.gnn.graph_pred_linear.parameters(), "lr":self.learning_rate})
        
        self.opt = optim.Adam(graph_params, lr=self.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        
        if self.baseline == 0:
            self.ckp_path_gnn = "checkpoints/checkpoints-GT/FS-GNNTR_GNN_tox21_10.pt"
            self.ckp_path_transformer = "checkpoints/checkpoints-GT/FS-GNNTR_Transformer_tox21_10.pt"
        elif self.baseline == 1:
            self.ckp_path_gnn = "checkpoints/checkpoints-baselines/GCN/checkpoint_gcn_gnn_tox21_5.pt"
                        
        # Model checkpoints:
        # GCN-Tox21-5-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_tox21_5.pt"
        # GCN-SIDER-5-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_sider_5.pt"    
        # GCN-Tox21-10-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_tox21_10.pt"
        # GCN-SIDER-10-GNN: "checkpoints/checkpoints-baselines/GCN/checkpoint_GCN_gnn_sider_10.pt"   
        
        # GraphSAGE-Tox21-5-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_tox21_5.pt"
        # GraphSAGE-SIDER-5-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_sider_5.pt"    
        # GraphSAGE-Tox21-10-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_tox21_10.pt"   
        # GraphSAGE-SIDER-10-GNN: "checkpoints/checkpoints-baselines/GraphSAGE/checkpoint_graphsage_gnn_sider_10.pt" 
        
        # GIN-Tox21-5-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_tox21_5.pt"
        # GIN-SIDER-5-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_sider_5.pt"
        # GIN-Tox21-10-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_tox21_10.pt"
        # GIN-SIDER-10-GNN: "checkpoints/checkpoints-baselines/GIN/checkpoint_GIN_gnn_sider_10.pt"
        
        self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
        self.transformer, self.meta_opt, start_epoch = load_ckp(self.ckp_path_transformer, self.transformer, self.meta_opt)
    
    def update_graph_params(self, loss, lr_update):
        grads = torch.autograd.grad(loss, self.gnn.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.gnn.parameters()) - parameters_to_vector(grads) * lr_update

    def meta_train(self):
        self.gnn.train()
        if self.baseline == 0:
            self.transformer.train()
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        
        support_sets, query_sets = sample_train(self.tasks, self.data, self.batch_size, self.n_support, self.n_query)
        
        for k in range(0, self.k_train):
            graph_params = parameters_to_vector(self.gnn.parameters())
            query_losses = torch.tensor([0.0]).to(device)
            for t in range(self.train_tasks):
                
                loss_support = torch.tensor([0.0]).to(device)
                loss_query = torch.tensor([0.0]).to(device)
                
                if self.baseline == 0:
                    inner_losses = torch.tensor([0.0]).to(device)
                    outer_losses = torch.tensor([0.0]).to(device) 
                    
                for batch_idx, batch in enumerate(tqdm(support_sets[t], desc="Iteration")):
                    batch = batch.to(device)
                    graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    label = batch.y.view(graph_pred.shape).to(torch.float64)
                    loss_graph = self.loss(graph_pred.double(), label)
                    support_loss = torch.sum(loss_graph)/graph_pred.size()[0]
                    loss_support += support_loss
                    
                    torch.cuda.empty_cache()    
                    
                    if self.baseline == 0:
                        pred, emb = self.transformer(self.gnn.pool(node_emb, batch.batch))
                        loss_tr = self.loss_transformer(F.sigmoid(pred).double(), label) 
                        inner_loss = torch.sum(loss_tr)/pred.size()[0]
                   
                    if self.baseline == 0:
                        inner_losses += inner_loss.item()
                    
                    del graph_pred, node_emb
                   
                updated_grad, updated_params = self.update_graph_params(loss_support, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())
                
                for batch_idx, batch in enumerate(tqdm(query_sets[t], desc="Iteration")):
                    batch = batch.to(device)
                    
                    graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    label = batch.y.view(graph_pred.shape).to(torch.float64)
                    
                    if self.baseline == 0:
                        logit, emb = self.transformer(self.gnn.pool(node_emb, batch.batch))
                        loss_tr = self.loss_transformer(F.sigmoid(logit).double(), label)
                        outer_loss = torch.sum(loss_tr)/logit.size()[0] 
                        
                    
                    loss_graph = self.loss(graph_pred.double(), label)
                    query_loss = torch.sum(loss_graph)/graph_pred.size()[0]
                    loss_query += query_loss
                    
                    del graph_pred, node_emb
                    
                    if self.baseline == 0:
                        outer_losses += outer_loss
              
                if t == 0:
                    query_losses = loss_query
                    if self.baseline == 0:
                        query_outer_losses = outer_losses
                else:
                    query_losses = torch.cat((query_losses, loss_query), 0)
                    if self.baseline == 0:
                        query_outer_losses =  torch.cat((query_outer_losses, outer_losses), 0)

                vector_to_parameters(graph_params, self.gnn.parameters())

            query_losses = torch.sum(query_losses)
            loss_graph = query_losses / self.train_tasks   
            loss_graph.to(device)
            
            if self.baseline == 0:
                query_outer_loss = torch.sum(query_outer_losses)
                loss_tr = query_outer_loss / self.train_tasks
                loss_tr.to(device) 
            
            self.opt.zero_grad()
            
            if self.baseline == 0:
                self.meta_opt.zero_grad()
            
            if self.baseline == 1:
                loss_graph.backward()
            
            if self.baseline == 0:
                loss_graph.backward(retain_graph=True)
                loss_tr.backward()
            
            self.opt.step()
            
            if self.baseline == 0:
                self.meta_opt.step()
            
        return 

    def meta_test(self):
        roc_scores = []
        t=0
        graph_params = parameters_to_vector(self.gnn.parameters())
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        
        for test_task in range(self.test_tasks):
            
            support_set, query_set = sample_test(self.tasks, test_task, self.data, self.batch_size, self.n_support, self.n_query)
            self.gnn.eval()
            if self.baseline == 0:
                self.transformer.eval()
                
            for k in range(0, self.k_test):
                
                graph_loss = torch.tensor([0.0]).to(device)
                
                if self.baseline == 0:
                    loss_logits = torch.tensor([0.0]).to(device)
                
                for batch_idx, batch in enumerate(tqdm(support_set, desc="Iteration")):
                    
                    batch = batch.to(device)
                    graph_pred, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(graph_pred.shape).to(torch.float64)
                    loss_graph = self.loss(graph_pred.double(), y)
                    graph_loss += torch.sum(loss_graph)/graph_pred.size()[0]
                    
                    if self.baseline == 0:
                        
                        with torch.no_grad():
                            val_logit, emb = self.transformer(self.gnn.pool(node_emb, batch.batch))
                        
                        loss_tr = self.loss_transformer(F.sigmoid(val_logit).double(), y)
                        loss_logits += torch.sum(loss_tr)/val_logit.size()[0] 
                          
                    del graph_pred, node_emb
                    
                updated_grad, updated_params = self.update_graph_params(graph_loss, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())
            
            torch.cuda.empty_cache()
            
            nodes=[]
            labels=[]
            y_label = []
            y_pred = []
           
            for batch_idx, batch in enumerate(tqdm(query_set, desc="Iteration")):
                batch = batch.to(device)
                
                with torch.no_grad(): 
                    logit, node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
                y_label.append(batch.y.view(logit.shape))
            
                if self.baseline == 0:
                    logit, node_emb = self.transformer(self.gnn.pool(node_emb, batch.batch))
                
                print(F.sigmoid(logit))
          
                pred = parse_pred(logit)
                
                node_emb_tsne = node_emb.cpu().detach().numpy() 
                y_tsne = batch.y.view(pred.shape).cpu().detach().numpy()
               
                for i in node_emb_tsne:
                    nodes.append(i)
                for j in y_tsne:
                    labels.append(j)
                
                y_pred.append(pred)   
                
            #t = plot_tsne(nodes, labels, t)
             
            roc_scores = roc_accuracy(roc_scores, y_label, y_pred)
               
            vector_to_parameters(graph_params, self.gnn.parameters())
        
        return roc_scores, self.gnn.state_dict(), self.transformer.state_dict(), self.opt.state_dict(), self.meta_opt.state_dict()
                
