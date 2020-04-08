import torch
import torch.nn as nn
import numpy as np
import time

class LinearModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(LinearModel, self).__init__()
        dim_list = [dim_in]
        dim_list += dim_hidden
        dim_list += [dim_out]
        self.linears = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i + 1]) for i in range(len(dim_hidden) + 1)])

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
            x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        return x


class Neural_LP_Learner(nn.Module):
    # take in x as tensor(relation, vx, vy)
    def __init__(self, option, matrix_kb):
        super(Neural_LP_Learner, self).__init__()
        self.seed = option.seed
        self.num_step = option.num_step
        self.num_layer = option.num_layer
        self.hidden_dim = option.rnn_state_size

        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout

        self.num_entity = option.num_entity
        self.num_relation = option.num_relation
        self.query_is_language = option.query_is_language

        self.num_query = option.num_query
        self.query_embed_size = option.query_embed_size

        torch.manual_seed(self.seed)

        self.LSTM_layer = torch.nn.LSTM(input_size=self.num_relation+self.num_entity*2, hidden_size=self.num_relation, num_layers=self.num_step)
        self.mem_att = None
        self.op_att = None
        self.matrix_kb = matrix_kb  # matrix: (num_relation, num_entity, num_entity)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.linear=torch.nn.Linear(self.num_relation,1)

    def forward(self, x):  # x: {relation, head, tail} in one-hot encoding
        batch_size = x.shape[0]
        memory = [torch.zeros(torch.Size((batch_size, self.num_entity))) for _ in range(self.num_step+1)]
        # memory: (T+1)*(batch,num_entity)
        input=torch.cat(tuple([x.unsqueeze(0) for _ in range(self.num_step)]))
        hidden_states, cell_layers=self.LSTM_layer(input)
        hidden_states = self.softmax(hidden_states)
        hidden_states.retain_grad()
        for i in range(batch_size):
            memory[0][i] = x[i, self.num_relation:self.num_relation+self.num_entity]  # vx=[0,0,...,1,...,0].T

        for i in range(self.num_step):
            temp_mat = torch.matmul(self.matrix_kb.T, self.softmax(hidden_states[i]).T).T  # matrix * attention: (
            # batch, num_entity, num_entity)
            temp_mem_att=[torch.zeros(i+1) for _ in range(batch_size)]

            prev_weighted_memory = torch.zeros((batch_size, self.num_entity))
            for j in range(batch_size):
                for k in range(i+1):
                    temp_mem_att[j][k]=torch.matmul(hidden_states[k][j],hidden_states[i][j])
                temp_mem_att[j]=self.softmax(temp_mem_att[j])
                for k in range(i+1):
                    prev_weighted_memory[j]+=temp_mem_att[j][k]*memory[k][j]

            # weighted previous memory: (num_entity, batch)
            memory[i+1] = torch.cat(
                tuple(torch.matmul(temp_mat[j], prev_weighted_memory[j, :]).unsqueeze(0) for j in range(batch_size)))
        self.mem_att=temp_mem_att
        self.op_att=self.softmax(hidden_states[-1])
        score = -torch.cat(tuple(torch.matmul(memory[-1][i].unsqueeze(0), x[i][self.num_relation+self.num_entity:self.num_relation+self.num_entity*2].unsqueeze(1)) for i in range(batch_size)))
        score.retain_grad()
        score = torch.sigmoid(score)
            #score: (batch)
        return score
