import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from mlp import MLP

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))

def get_laplacian(A):
    D = A.sum(dim=1) + 1e-6
    D = 1 / torch.sqrt(D)
    D = torch.diag(D)
    L = -D@A@D
    return L

def get_normalized_adjacency(A):
    D = A.sum(dim=1) + 1e-6
    D = 1 / torch.sqrt(D)
    D = torch.diag(D)
    A = D@A@D
    return A

def get_edge_prob(logits, gumbel_noise=False, beta=0.5, hard=False):
    if gumbel_noise:
        y = logits + sample_gumbel(logits.size()).to(logits.device)
    else:
        y = logits
    edge_prob_soft = torch.softmax(beta * y, dim=0)

    if hard:
        _, edge_prob_hard = torch.max(edge_prob_soft.data, dim=0)
        edge_prob_hard = F.one_hot(edge_prob_hard)
        edge_prob_hard = edge_prob_hard.permute(1,0)
        edge_prob = edge_prob_hard - edge_prob_soft.data + edge_prob_soft
    else:
        edge_prob = edge_prob_soft
    return edge_prob

class A_MSG(nn.Module):


    def __init__(self, num_vertex, h_dim, hidden_dim, num_hidden, act):
        super(A_MSG, self).__init__()

        self.num_vertex = num_vertex
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden

        # A model
        self.a_inter_act_fun_head0 = MLP(dim_in=2 * h_dim, dim_out=h_dim, num_hidden=num_hidden, sigma= 0.01, activation = act)
        self.a_inter_act_fun_head1 = MLP(dim_in=2 * h_dim, dim_out=h_dim, num_hidden=num_hidden, sigma=0.01, activation = act)
        self.a_rou = MLP(dim_in=2 * h_dim, dim_out=h_dim, num_hidden=num_hidden, sigma= 0.01, activation = act)

        edges = np.ones(num_vertex) - np.zeros([num_vertex,num_vertex])
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge_index = torch.tensor([np.array(self.send_edges), np.array(self.recv_edges)]).cuda()


    def forward(self, A_H, edge_probs, batch_event_idx):

        batch_size = A_H.shape[0]
        batch_idx = torch.arange(batch_size).to(A_H.device)

        source_A_H = A_H[batch_idx, batch_event_idx].unsqueeze(dim=1).repeat(1, self.num_vertex, 1)
        edge = torch.cat((source_A_H, A_H), dim=-1)
        agg_node = A_H.clone()

        all_send_edge_id = np.array([])
        for e_i, event in enumerate(batch_event_idx):
            send_edge_id = np.where(self.send_edges == event)[0]
            all_send_edge_id = np.concatenate([all_send_edge_id, send_edge_id])

        send_edge_prob_0 = edge_probs[0, all_send_edge_id].contiguous().view(batch_size, self.num_vertex, 1)
        send_edge_prob_1 = edge_probs[1, all_send_edge_id].contiguous().view(batch_size, self.num_vertex, 1)

        #Correct
        agg_msg_A = send_edge_prob_0 * self.a_inter_act_fun_head0(edge) + send_edge_prob_1 * \
                    self.a_inter_act_fun_head1(edge)

        update_A_H = self.a_rou(torch.cat((agg_node, agg_msg_A), dim=-1))
        update_A_H[batch_idx, batch_event_idx] = 0.0
        A_H = A_H + update_A_H

        return A_H



