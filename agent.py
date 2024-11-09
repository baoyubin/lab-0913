import random

import numpy as np
import torch
import torch.nn as nn

##https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network

class Replaymemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.bus_num = 10
        self.MEMORY_SIZE = 10**5
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

    @torch.no_grad()
    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self(obs_tensor.unsqueeze(0))
        max_q_value = torch.argmax(input=q_value)
        action = max_q_value.detach().item()
        #TODO 转二进制
        action_bin = bin(action)[2:]
        action = np.array(list(action_bin.rjust(10, '0')), dtype=np.int32)
        return action


import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self(state)
        bool_mask = action > 0.5
        #
        # # 然后使用布尔张量来创建一个新的张量，其中True被替换为1.0（或你想要的任何浮点值），False被替换为0.0
        # # 这里我们直接转换为浮点类型，因为布尔张量在转换为浮点张量时会自动变为0.0和1.0
        # action2_float = bool_mask.float()
        #
        # # 如果你需要整数类型的张量，可以进一步转换
        action = action.int()
        return action
        # return action.detach().cpu().numpy()[0, 0]

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Agent:
    def __init__(self, n_input, n_output, GAMA, learning_race):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMA = GAMA
        self.learning_race = learning_race
       ## self.training_interval = 10

        self.memo = Replaymemory(self.n_input, self.n_output)
        self.target_actor = Actor(self.n_input, self.n_output)
        self.actor = Actor(self.n_input, self.n_output)
        self.target_critic = Critic(self.n_input, self.n_output)
        self.critic = Critic(self.n_input, self.n_output)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_race)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_race)
    @torch.no_grad()
    def predict_action(self, state):
        ''' 用于预测，不需要计算梯度
        '''
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.cpu().numpy()[0, 0]