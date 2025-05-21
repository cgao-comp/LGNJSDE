import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
import tqdm
from graph_modules import A_MSG, get_edge_prob
from sde_modules import SDEFunc, LNSDEFunc
import torch
import numpy as np
import random
import os

class LGNJSDE(nn.Module):

    def __init__(self, num_vertex, h_dim, hidden_dim, num_hidden, act=nn.Tanh(), num_divide=10,
                 device = None, func_type = 'sde'):

        super(LGNJSDE, self).__init__()

        self.num_vertex = num_vertex
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.num_divide = num_divide

        # init the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device
        else:
            self.device = device

        self.func_type = func_type

        if func_type == 'sde':
            self.func = SDEFunc(dim_h=h_dim, num_vertex=num_vertex, dim_hidden=hidden_dim, num_hidden=num_hidden,
                                activation=act).to(device)
        elif func_type == 'lnsde':
            self.func = LNSDEFunc(dim_h=h_dim, num_vertex=num_vertex, dim_hidden=hidden_dim, num_hidden=num_hidden,
                                activation=act).to(device)
        else:
            raise Exception("this type of func dose not exsit")


        self.register_parameter('logits',nn.Parameter(torch.zeros([2, num_vertex * num_vertex], device=device),
                                                 requires_grad=True))

        self.MSG = A_MSG(num_vertex, h_dim, hidden_dim, 1, act).to(device) # Avoid too deep

        self.register_parameter('h0', nn.Parameter(
            torch.empty([num_vertex, h_dim], device=device).normal_(mean=0, std=0.01).requires_grad_()))

        edges = np.ones(num_vertex) - np.zeros([num_vertex, num_vertex])
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge_index = torch.tensor([np.array(self.send_edges), np.array(self.recv_edges)]).to(device)

    def forward(self, batch_train_time, batch_train_type, batch_train_mask):
        # [batch_size, num_edges, seq, 2]
        batch_size, seq_len = batch_train_time.shape

        self.edge_prob = get_edge_prob(self.logits, False, 0.5, False)#

        batch_h0 = self.h0.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        a_nll, a_l_batch_l = self.train_forward_pass(batch_h0, batch_train_time, batch_train_type,
                                                      batch_train_mask, batch_size)

        return a_nll, a_l_batch_l

    def training_step(self, batch):

        batch_train_time, batch_train_type, batch_train_mask = batch
        a_nll, a_l_batch_l = self.forward(batch_train_time.to(self.device),
                           batch_train_type.to(self.device), batch_train_mask.to(self.device))

        return a_nll

    def validation_step(self, batch):

        batch_train_time, batch_train_type, batch_train_mask = batch

        num_event = torch.sum(batch_train_mask)

        a_nll, a_l_batch_l = self.forward(batch_train_time.to(self.device), batch_train_type.to(self.device),
                           batch_train_mask.to(self.device))


        return a_nll, num_event

    def configure_optimizers(self, lr = 1e-3):


        optimizer = optim.Adam([{'params': self.func.parameters()},
                                    {'params': self.MSG.parameters()},
                                    {'params': self.h0, 'lr': 1e-1, 'weight_decay': 1e-5},
                                    {'params': self.logits, 'lr': 1e-1, 'weight_decay': 1e-5},
                                    ], lr=lr, weight_decay=1e-5)

        return optimizer

    def EulerSolver(self, a_h_initial, adjacent_events, num_divide = None):

        if num_divide is None:
            num_divide = self.num_divide

        dt = torch.diff(adjacent_events, dim=1) / num_divide
        ts = torch.cat([adjacent_events[:, 0].unsqueeze(dim=1) + dt * j for j in range(num_divide + 1)], dim=1)

        #only consider A
        a_l_ts = torch.Tensor().to(a_h_initial.device)
        a_l_initial = self.func.e(a_h_initial)
        a_l_ts = torch.cat((a_l_ts, a_l_initial.unsqueeze(2)), dim=2)
        h_dt = dt.unsqueeze(dim=1).repeat(1, self.num_vertex, 1)
        h_last = adjacent_events[:, [0]].unsqueeze(dim=1).repeat(1, self.num_vertex, 1)
        
        for i in range(num_divide):
            h_diff_t = h_dt * (i + 1)
            # update hidden
            # [h, s - t_i, t_i]
            a_h_output = (a_h_initial + self.func.f(a_h_initial, h_diff_t, h_last) * h_dt +
                          self.func.g(a_h_initial, h_diff_t) * torch.sqrt(h_dt) * torch.randn_like(a_h_initial))

            # transform h to eta
            a_l_output = self.func.e(a_h_output.clone())
            a_l_ts = torch.cat((a_l_ts, a_l_output.unsqueeze(2)), dim=2)
            a_h_initial = a_h_output
            a_l_initial = a_l_output

        return ts, a_l_ts, a_h_initial

    def jump_and_msg_passing(self, A_H, batch_idx, event_idx):
        '''
        :param sde, filter
        :param h: [batch, n_marks, h_dim]
        :param event_idx: [batch, ] 在这一时刻发生的事件
        :param posterior_edge_probs: [batch, n_edges, 2] softmax
        :return: update_h : h : [batch, n_marks, h_dim], edge_probs_kl [batch, ]
        '''
        # 1. Update h_i for the event has occured

        # First Update Other Hidden State
        A_H = self.MSG(A_H, self.edge_prob, event_idx)
        # Update the Event
        A_H[batch_idx, event_idx] = A_H[batch_idx, event_idx] + self.func.h(A_H[batch_idx, event_idx])

        return A_H

    def train_forward_pass(self, h0, time_seqs, type_seqs, mask, batch_size):
        '''
        :param model: A Graph ODE model
        :param h0: initial h0 [batch_size, n_marks, h_dim]
        :param time_seqs:
        :param type_seqs:
        :param mask:
        :param device:
        :param batch_size:
        :param dim_eta: n_marks
        :param num_divide: the number of time divided in a time interval
        :param posterior_edge_prob_logits : [batch, n_edges, seq, 2]
        '''

        a_h0 = h0.clone()

        padded_seq_length = len(time_seqs[0])

        # eta_batch_l[i,j,:] is the left-limit of \mathbf{lambda} at the j-th event in the i-th sequence
        a_l_batch_l = torch.zeros(batch_size, padded_seq_length, self.num_vertex, device=self.device)
        a_l_batch_r = torch.zeros(batch_size, padded_seq_length, self.num_vertex, device=self.device)

        # eta_time_l[i,j] == eta_batch_l[i,j,:][type_seqs[i,j]]
        a_l_time_l = torch.zeros(batch_size, padded_seq_length, device=self.device)
        a_l_time_r = torch.zeros(batch_size, padded_seq_length, device=self.device)

        # before the first jump
        a_l0 = self.func.e(a_h0.clone())
        a_l_batch_l[:, 0, :] = a_l0

        # first jump
        batch_idx = torch.arange(batch_size)
        event_type = type_seqs[:, 0].tolist()
        a_l_time_l[:, 0] = a_l_batch_l[list(range(0, batch_size)), 0, event_type]

        # eta(t+) = eta(t-) + h(eta, t)
        # first jump
        a_h0 = self.jump_and_msg_passing(a_h0, batch_idx, event_type)

        a_l_batch_r[:, 0, :] = self.func.e(a_h0)
        a_l_time_r[:, 0] = a_l_batch_r[list(range(0, batch_size)), 0, event_type]

        tsave = torch.Tensor().to(self.device)
        a_l_tsave = torch.Tensor().to(self.device)

        for i in range(padded_seq_length - 1):

            adjacent_events = time_seqs[:, i:i + 2]

            ts, a_l_ts_l, a_h0 = self.EulerSolver(a_h0, adjacent_events)

            tsave = torch.cat((tsave, ts), dim=1)
            a_l_tsave = torch.cat((a_l_tsave, a_l_ts_l), dim=2)

            a_l_batch_l[:, i + 1, :] = a_l_ts_l[:, :, -1]

            a_l_ts_r = a_l_ts_l.clone()
            event_type = type_seqs[:, i + 1].tolist()

            a_h0 = self.jump_and_msg_passing(a_h0, batch_idx, event_type)

            a_l_ts_r[:, :, -1] = self.func.e(a_h0)

            a_l_batch_r[:, i + 1, :] = a_l_ts_r[:, :, -1]
            a_l_time_l[:, i + 1] = a_l_batch_l[list(range(0, batch_size)), i + 1, event_type]
            a_l_time_r[:, i + 1] = a_l_batch_r[list(range(0, batch_size)), i + 1, event_type]
            a_l_initial = a_l_ts_r[:, :, -1]

        a_masked_l_time_l = torch.log(a_l_time_l + 1e-16) * mask  # lambda --> log(lambda)
        a_sum_term = torch.sum(a_masked_l_time_l)

        mask_without_first_col = mask[:, 1:]

        expanded_mask = mask_without_first_col.unsqueeze(2).repeat(1, 1, self.num_divide + 1).view(batch_size, -1)
        expanded_mask = expanded_mask.unsqueeze(1).repeat(1, self.num_vertex, 1)

        a_l_tsave = a_l_tsave * expanded_mask  # mask the eta_tsave

        expanded_diff_tsave = torch.diff(tsave).unsqueeze(1).repeat(1, self.num_vertex, 1)

        a_integral_term = torch.sum(
            (a_l_tsave[:, :, :-1] * expanded_mask[:, :, :-1] + a_l_tsave[:, :, 1:] * expanded_mask[:, :, 1:]) * (
                    expanded_diff_tsave * expanded_mask[:, :, 1:])) / 2  # reason for mask: e^0=1

        a_nll = a_integral_term - a_sum_term

        return a_nll, a_l_batch_l

    def predict_forward_pass(self, h0, time_seqs, type_seqs, mask, batch_size):

        a_h0 = h0.clone()

        padded_seq_length = len(time_seqs[0])

        a_l_batch_l = torch.zeros(batch_size, padded_seq_length, self.num_vertex, device=self.device)
        a_l_batch_r = torch.zeros(batch_size, padded_seq_length, self.num_vertex, device=self.device)

        a_h_batch_l = torch.zeros(batch_size, padded_seq_length, self.num_vertex, self.h_dim, device=self.device)
        a_h_batch_r = torch.zeros(batch_size, padded_seq_length, self.num_vertex, self.h_dim, device=self.device)

        a_l_time_l = torch.zeros(batch_size, padded_seq_length, device=self.device)
        a_l_time_r = torch.zeros(batch_size, padded_seq_length, device=self.device)

        a_l0 = self.func.e(a_h0)
        a_l_batch_l[:, 0, :] = a_l0
        a_h_batch_l[:, 0, :, :] = a_h0

        batch_idx = torch.arange(batch_size)
        event_type = type_seqs[:, 0].tolist()
        a_l_time_l[:, 0] = a_l_batch_l[list(range(0, batch_size)), 0, event_type]

        a_h0 = self.jump_and_msg_passing(a_h0, batch_idx, event_type)
        a_l_batch_r[:, 0, :] = self.func.e(a_h0)
        a_h_batch_r[:, 0, :, :] = a_h0
        a_l_time_r[:, 0] = a_l_batch_r[list(range(0, batch_size)), 0, event_type]

        tsave = torch.Tensor().to(self.device)
        a_l_tsave = torch.Tensor().to(self.device)

        for i in range(padded_seq_length - 1):
            adjacent_events = time_seqs[:, i:i + 2]
            ts, a_l_ts_l, a_h0 = self.EulerSolver(a_h0, adjacent_events)
            tsave = torch.cat((tsave, ts), dim=1)
            a_l_tsave = torch.cat((a_l_tsave, a_l_ts_l), dim=2)

            a_l_batch_l[:, i + 1, :] = a_l_ts_l[:, :, -1]
            a_h_batch_l[:, i + 1, :, :] = a_h0

            a_l_ts_r = a_l_ts_l.clone()
            event_type = type_seqs[:, i + 1].tolist()

            a_h0 = self.jump_and_msg_passing(a_h0, batch_idx, event_type)

            a_l_ts_r[:, :, -1] = self.func.e(a_h0)
            a_l_batch_r[:, i + 1, :] = a_l_ts_r[:, :, -1]
            a_h_batch_r[:, i + 1, :, :] = a_h0
            a_l_time_l[:, i + 1] = a_l_batch_l[list(range(0, batch_size)), i + 1, event_type]
            a_l_time_r[:, i + 1] = a_l_batch_r[list(range(0, batch_size)), i + 1, event_type]
            a_l_initial = a_l_ts_r[:, :, -1]

        a_masked_l_time_l = torch.log(a_l_time_l + 1e-16) * mask
        a_sum_term = torch.sum(a_masked_l_time_l)

        mask_without_first_col = mask[:, 1:]
        expanded_mask = mask_without_first_col.unsqueeze(2).repeat(1, 1, self.num_divide + 1).view(batch_size, -1)
        expanded_mask = expanded_mask.unsqueeze(1).repeat(1, self.num_vertex, 1)

        a_l_tsave = a_l_tsave * expanded_mask

        expanded_diff_tsave = torch.diff(tsave).unsqueeze(1).repeat(1, self.num_vertex, 1)

        a_integral_term = torch.sum(
            (a_l_tsave[:, :, :-1] * expanded_mask[:, :, :-1] + a_l_tsave[:, :, 1:] * expanded_mask[:, :, 1:]) * (
                    expanded_diff_tsave * expanded_mask[:, :, 1:])) / 2

        a_nll = a_integral_term - a_sum_term

        return  a_l_batch_l, a_l_batch_r, a_h_batch_l, a_h_batch_r, a_nll

    def predict(self, time_seqs, type_seqs, h, n_samples, device):

        estimate_dt = []
        next_dt = []
        error_dt = []
        estimate_type = []
        next_type = []
        loss_list = []

        self.device = device

        for idx, seq in tqdm.tqdm(enumerate(time_seqs)):

            seq_time = seq.unsqueeze(0)
            seq_type = type_seqs[idx].unsqueeze(0)
            mask = torch.ones(1, len(seq), device=self.device)

            h0 = self.h0.clone().unsqueeze(dim=0)

            l_seq_l, l_seq_r, h_seq_l, h_seq_r, loss_idx = self.predict_forward_pass(h0, seq_time, seq_type,
                                                                                mask, batch_size=1)

            dt_seq = torch.diff(seq)
            max_dt = torch.max(dt_seq)

            estimate_seq_dt = []
            next_seq_dt = dt_seq.tolist()
            error_seq_dt = []
            estimate_seq_type = []
            next_seq_type = type_seqs[idx][1:].tolist()

            for i in range(len(seq) - 1):
                last_t = seq[i]
                n_dt = dt_seq[i]
                timestep = h * max_dt / n_samples
                tau = last_t + torch.linspace(0, h * max_dt, n_samples).to(self.device)
                d_tau = tau - last_t

                h_last_t = h_seq_r[:, i, :, :]

                adjacent_events = torch.tensor([last_t, last_t + h * max_dt], device=self.device).unsqueeze(0)

                _, l_tau, _ = self.EulerSolver(h_last_t, adjacent_events, n_samples - 1)

                l_tau = l_tau.squeeze(0)

                intens_tau = l_tau
                intens_tau_sum = intens_tau.sum(dim=0)
                integral = torch.cumsum(timestep * intens_tau_sum, dim=0)

                density = intens_tau_sum * torch.exp(-integral)
                d_tau_f_tau = d_tau * density

                e_dt = (timestep * 0.5 * (d_tau_f_tau[1:] + d_tau_f_tau[:-1])).sum()
                err_dt = torch.abs(e_dt - n_dt)
                e_type = torch.argmax(l_seq_l[0][i + 1])

                estimate_seq_dt.append(e_dt.item())
                error_seq_dt.append(err_dt.item())
                estimate_seq_type.append(e_type.item())

            loss_list.append(loss_idx)
            estimate_dt.extend(estimate_seq_dt)
            next_dt.extend(next_seq_dt)
            error_dt.extend(error_seq_dt)
            estimate_type.extend(estimate_seq_type)
            next_type.extend(next_seq_type)

        error_dt_tensor = torch.tensor(error_dt)
        RMSE = np.linalg.norm(error_dt_tensor.detach().numpy(), ord=2) / (len(error_dt_tensor) ** 0.5)
        acc = accuracy_score(next_type, estimate_type)
        f1 = f1_score(next_type, estimate_type, average='weighted')
        loss = sum(loss_list)

        return loss, RMSE, acc, f1

def train(model, max_epochs, train_dataloader, val_dataloader, seed, data_name, log, lr = 1e-3, impatience = 20):

    func_type = model.func_type

    print("Func : {}, Seed {} start !!!!!!".format(func_type, seed))
    print("Func : {}, Seed {} start !!!!!!".format(func_type, seed), flush=True, file=log)

    best_nll = np.inf
    count = 0

    optimizer = model.configure_optimizers(lr=lr)

    for e_i in tqdm.tqdm(range(max_epochs)):
        # Train Model
        tot_a_ll = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            a_nll = model.training_step(batch)
            a_nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_a_ll = tot_a_ll + a_nll.detach().cpu()

        val_a_ll = 0.0
        val_num_event = 0.0

        model.eval()
        for batch in val_dataloader:

            a_nll, num_event = model.validation_step(batch)
            val_a_ll += a_nll.detach().cpu().item()
            val_num_event += num_event

        val_a_ll = val_a_ll / val_num_event
        print("Epoch : {}, A train nll : {}, A val nll : {}".format(e_i, tot_a_ll, val_a_ll), flush=True, file=log)

        # Early Stop
        if val_a_ll < best_nll:
            count = 0
            best_nll = val_a_ll
            #Save Model
            torch.save(model, './ckpt/{}-{}-{}-model.pth'.format(data_name, func_type, seed))

        else:
            count += 1
            if count > impatience:
                print("Early Stop at {} -------- Best A val nll : {}".format(e_i, best_nll))
                break
