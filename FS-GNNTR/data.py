import os
import sys
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
import random

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def datasets(data):
    if data == "tox21":
        return [[6956, 309],[6521, 237],[5781, 768],[5521, 300],[5400, 793],[6605, 350],[6264, 186],[4890, 942],[6808, 264],[6095, 372],[4892, 918],[6351, 423]]
    elif data == "sider":       
        return [[684, 743],[431, 996],[1405, 22],[551, 876],[276, 1151],[430, 997],[129, 1298],[1176, 281],[403, 1024],[700, 727],[1051, 376],[135, 1292],[1104, 323],[1214, 213],[319, 1108],[542, 885],[109, 1318],[1174, 253],[421, 1006],[367, 1060],[411, 1016],[516, 911],[1302, 125],[768, 659],[439, 988],[123, 1304],[481, 946]]
    elif data == "muv":
        return [[14814, 27], [14705, 29], [14698, 30], [14593, 30], [14873, 29], [14572, 29], [14614, 30], [14383, 28], [14807, 29], [14654, 28], [14662, 29], [14615, 29], [14637, 30], [14681, 30], [14622, 29], [14745, 29], [14722, 24]]

def random_sampler(D, d, t, k, n, train):
    
    data = datasets(d)
    
    s_pos = random.sample(range(0,data[t][0]), k)
    s_neg = random.sample(range(data[t][0],len(D)), k)
    s = s_pos + s_neg
    random.shuffle(s)

    samples = [i for i in range(0, len(D)) if i not in s]
    
    if train == True:     
        q = random.sample(samples, n)
    else:
        random.shuffle(samples)
        q = samples

    S = D[torch.tensor(s)]
    Q = D[torch.tensor(q)]

    return S, Q


def split_into_directories(data):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    file = open(os.path.join(root_dir, 'Data/{}/original/{}.csv'.format(data,data)), 'r').readlines()[1:]
    np.random.shuffle(file)
    
    T = {}
  
    if data == "tox21":
        for k,j  in enumerate(file):
            sample = j.split(",")
           
            for i in range(12):
                if i not in T:
                    T[i] = [[],[]]
                if sample[i] == "0":
                    T[i][0].append(sample[13].rstrip())
                elif sample[i] == "1":
                    T[i][1].append(sample[13].rstrip())
      
        for t in T:
            d = 'Data/' + data + "/pre-processed/" + "task_" +str(t+1)
            os.makedirs(d, exist_ok=True)
            os.makedirs(d + "/raw", exist_ok=True)
            os.makedirs(d + "/processed", exist_ok=True)
            
            with open(d + "/raw/" + data + "_task_" + str(t+1), "wb") as fp:   #Pickling
                pickle.dump(T[t], fp)
    
    if data == "sider":
        for k,j  in enumerate(file):
            sample = j.split(",")
            smile = sample[0].rstrip()
            sample = sample[1:]
            for i in range(27):
                
                if i not in T:
                    T[i] = [[],[]]
                if sample[i] == "0" or sample[i] == "0\n":
                    T[i][0].append(smile)
                elif sample[i] == "1" or sample[i] == "1\n":
                    T[i][1].append(smile)
      
        for t in T:
            d = 'Data/' + data + "/pre-processed/" + "task_" +str(t+1)
            os.makedirs(d, exist_ok=True)
            os.makedirs(d + "/raw", exist_ok=True)
            os.makedirs(d + "/processed", exist_ok=True)
            
            with open(d + "/raw/" + data + "_task_" + str(t+1), "wb") as fp:   #Pickling
                pickle.dump(T[t], fp)
        
    if data == "muv":
        for k,j  in enumerate(file):
            sample = j.split(",")
           
            for i in range(17):
                if i not in T:
                    T[i] = [[],[]]
                if sample[i] == "0" or sample[i] == "\"0":
                    T[i][0].append(sample[18].rstrip())
                elif sample[i] == "1" or sample[i] == "\"1":
                    T[i][1].append(sample[18].rstrip())
      
        for t in T:
            d = 'Data/' + data + "/pre-processed/" + "task_" +str(t+1)
            os.makedirs(d, exist_ok=True)
            os.makedirs(d + "/raw", exist_ok=True)
            os.makedirs(d + "/processed", exist_ok=True)
            
            with open(d + "/raw/" + data + "_task_" + str(t+1), "wb") as fp:   #Pickling
                pickle.dump(T[t], fp)
 
            
        

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                
        elif self.dataset == 'muv':
             smiles_list, rdkit_mol_objs, labels = \
                 _load_muv_dataset(self.raw_paths[0])
             for i in range(len(smiles_list)):
                 print(i)
                 rdkit_mol = rdkit_mol_objs[i]
                 data = mol_to_graph_data_obj_simple(rdkit_mol)
                 # manually add mol id
                 data.id = torch.tensor(
                     [i])  # id here is the index of the mol in
                 # the dataset
                 data.y = torch.tensor(labels[i, :])
                 data_list.append(data)
                 data_smiles_list.append(smiles_list[i])


        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'processed.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset

def create_circular_fingerprint(mol, radius, size, chirality):
    """
    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)


def _load_tox21_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    with open(input_path, 'rb') as f:
       binary_list = pickle.load(f)

    print(binary_list)
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels

def _load_muv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    with open(input_path, 'rb') as f:
       binary_list = pickle.load(f)

    print(binary_list)
  
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i[:-1])
    
    rdkit_mol_objs_list = []
    for s in smiles_list:
        rdkit_mol_objs_list.append(Chem.MolFromSmiles(s))
        
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 
    
    

    return smiles_list, rdkit_mol_objs_list, labels

def _load_sider_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """

    with open(input_path, 'rb') as f:
       binary_list = pickle.load(f)

    print(binary_list)
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def dataset(data):
    root = "Data/" + data + "/pre-processed/"
    T = 17
    if data == "sider":
        T = 27
    elif data == "tox21":
        T = 12
    elif data == "muv":
        T = 27

    for task in range(T):
        build = MoleculeDataset(root + "task_" + str(task+1), dataset=data)

if __name__ == "__main__":
    # split data in mutiple data directories and convert to RDKit.Chem graph structures
    #split_into_directories("muv")
    dataset("muv")
