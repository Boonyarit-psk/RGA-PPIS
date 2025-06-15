import pickle
import math
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import time

import warnings
warnings.filterwarnings("ignore")

# Feature Path
Feature_Path = "Feature/"

# Seed
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
BASE_MODEL_TYPE = 'AGAT' 
ADD_NODEFEATS = 'all'  
USE_EFEATS = False  

if BASE_MODEL_TYPE == 'GCN':
    USE_EFEATS = False

MAP_CUTOFF = 14 
DIST_NORM = 15 

# INPUT_DIM
if ADD_NODEFEATS == 'all': 
    INPUT_DIM = 54 + 7 + 1 
elif ADD_NODEFEATS == 'atom_feats':  
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54

HIDDEN_DIM = INPUT_DIM 
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5
BATCH_SIZE = 1
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0 
NUM_CLASSES = 2  
LAYER = 8 
NUMBER_EPOCHS = 150
NUMBER_FREEZE = HIDDEN_DIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)

def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def normalizefortensor(mx):
    rowsum = torch.sum(mx, dim=1)
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)
    result = r_mat_inv.float() @ mx.float() @ r_mat_inv.float()
    return result

def cal_edges(sequence_name, radius=MAP_CUTOFF): 
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius)) 
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1) 
    radius_index_list = [list(nodes) for nodes in radius_index_list] 
    return radius_index_list 

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = adjacency_matrix.astype(np.float32)
    #norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix

def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix, pos = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, pos


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='Feature/psepos/Train335_psepos_SC.pkl'):
        
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        node_features = torch.from_numpy(node_features)

        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
            node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)
        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )
        
        adj_matrix = load_graph(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

        return sequence_name, sequence, label, node_features, G, adj_matrix, pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list
    
    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)

import torch
from torch import nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int, num_hops=4):
        super(GraphAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.n_hidden = out_features
        self.num_hops = num_hops
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))
        self.layer_norm = nn.LayerNorm(out_features)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, pos: torch.Tensor):
        n_nodes = h.shape[0]
        h_transformed = torch.matmul(h, self.W)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        e = self._get_attention_scores(h_transformed, pos)
        connectivity_mask = -9e15 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask)
        attention = F.softmax(e, dim=-1)
        h_prime = torch.matmul(attention, h_transformed)
        h_prime = self.layer_norm(h_prime.mean(dim=0))
        h_prime = h + h_prime
        pos_updated = self.update_positions(pos, h_prime, attention)

        return h_prime, pos_updated
    
    def _get_attention_scores(self, h_transformed: torch.Tensor, pos: torch.Tensor):#, adj_mat: torch.Tensor):
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])
        distance_scores = self.compute_distance_scores(pos)
        e = source_scores + target_scores.permute(0, 2, 1) - distance_scores
        return self.leakyrelu(e)

    def compute_distance_scores(self, pos: torch.Tensor):
        return torch.cdist(pos, pos, p=2)  # Euclidean distance
    
    def update_positions(self, pos: torch.Tensor, h_prime: torch.Tensor, attention: torch.Tensor):
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        attention_sum = attention.mean(dim=0).unsqueeze(-1)
        delta_pos = torch.sum(attention_sum * rel_pos, dim=1)
        pos_updated = pos + delta_pos

        return pos_updated

class deepGAT(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, lamda, alpha, dropout, n_heads = 2):
        super(deepGAT, self).__init__()

        self.baseModules = nn.ModuleList()
        for _ in range(nlayers):
            self.baseModules.append(GraphAttentionLayer(nfeat, nhidden, n_heads))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nhidden, 128))
        self.fcs.append(nn.Linear(128, 32))
        self.fcs.append(nn.Linear(32, nclass))
        self.act_fn = nn.ReLU()
        self.act_fns = nn.ELU()
        self.dropout = dropout
        
    def forward(self, x, pos, adj_matrix=None, graph=None, efeats=None, add_features=None):

        layer_inner = x
        new_position = pos

        for a, baseMod in enumerate(self.baseModules):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner, new_position = baseMod(layer_inner, adj_matrix, new_position)
            layer_inner = self.act_fns(layer_inner)

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](layer_inner))
        layer_inner = self.act_fn(self.fcs[1](layer_inner))
        layer_inner = self.fcs[-1](layer_inner)

        return layer_inner


class GATPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(GATPPIS, self).__init__()

        self.deep_agat = deepGAT(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, nclass=nclass,
                                  dropout=dropout, lamda=lamda, alpha=alpha)

        self.criterion = nn.CrossEntropyLoss()  # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10, min_lr=1e-6)

    def forward(self, x, graph, adj_matrix, pos, add_features=None):
        x = x.float()
        x = x.view([x.shape[0]*x.shape[1], x.shape[2]])

        output = self.deep_agat(x=x, adj_matrix=adj_matrix, graph=graph, efeats=graph.edata['ex'], pos = pos, add_features = add_features)
        
        return output