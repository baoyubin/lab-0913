import random

import numpy as np
import torch
import torch.nn as nn

from ReplayTree import ReplayTree


##https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network

class Replaymemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.bus_num = 10
        self.MEMORY_SIZE = 2000
        self.BATCH_SIZE = 360
        #TODO
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.int32) #TODO
        self.all_a = np.empty(shape=(self.MEMORY_SIZE, self.bus_num), dtype=np.int32)
        self.all_r = np.empty(shape=self.MEMORY_SIZE, dtype=np.float32)
        self.all_done = np.empty(shape=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.int32) #TODO
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self, s, a, r, done, s_):
        self.all_s[self.t_memo] = s
        self.all_a[self.t_memo] = a
        self.all_r[self.t_memo] = r
        self.all_done[self.t_memo] = done
        self.all_s_[self.t_memo] = s_
        self.t_max = max(self.t_memo + 1, self.t_max)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):

        if self.t_max >= self.BATCH_SIZE:
            idxes = random.sample(range(0, self.t_max), self.BATCH_SIZE)
        else:
            idxes = range(0, self.t_max)
        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.uint8).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor

class Dqn(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, n_output)
        )


    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self(obs_tensor.unsqueeze(0))
        max_q_value = torch.argmax(input=q_value)
        oraction = max_q_value.detach().item()
        #TODO 转二进制
        action_bin = bin(oraction)[2:]
        action = np.array(list(action_bin.rjust(10, '0')), dtype=np.int32)
        return action, oraction


class Agent:
    def __init__(self, n_input, n_output, GAMA, learning_race):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMA = GAMA
        self.learning_race = learning_race
         ## self.training_interval = 10

        self.memo = ReplayTree(2048)
        self.target_net = Dqn(self.n_input, self.n_output)
        self.online_net = Dqn(self.n_input, self.n_output)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_race)
