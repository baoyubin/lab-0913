import pickle
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def plot_reward(self, isSmooth=False, window_size=5, polyorder=3,other = False, topk = False):
    from more_itertools import chunked
    smoothed_list = [sum(x)  for x in chunked(self.reward_list, self.T+1)] ##一轮迭代的总奖励
    smoothed_top_list = [sum(x) for x in chunked(self.rw_topk, self.T+1)]
    smoothed_close_cost = [sum(x) for x in chunked(self.rw_all_0, self.T+1)] ##一轮迭代的总奖励
    smoothed_open_cost = [sum(x) for x in chunked(self.rw_all_1, self.T+1)] ##一轮迭代的总奖励
    if isSmooth:
        # # 定义平滑参数，例如窗口大小和多项式阶数
        # window_size = 5  # 可根据实际情况调整
        # polyorder = 3  # 通常应小于窗口大小的一半

        # 应用平滑
        smoothed_list = savgol_filter(smoothed_list, window_size, polyorder)
        smoothed_top_list = savgol_filter(smoothed_top_list, window_size, polyorder)
        smoothed_close_cost = savgol_filter(smoothed_close_cost, window_size, polyorder)
        smoothed_open_cost = savgol_filter(smoothed_open_cost, window_size, polyorder)
    plt.plot(np.arange(len(smoothed_list)), smoothed_list, color='b', label='dqn')
    if other:
        plt.plot(np.arange(len(smoothed_close_cost)), smoothed_close_cost, color='g', label='all_0')
        plt.plot(np.arange(len(smoothed_open_cost)), smoothed_open_cost, color='y', label='all_1')
        if topk:
            plt.plot(np.arange(len(smoothed_top_list)), smoothed_top_list, color='r', label='topk')
    plt.title("idle_w" + str(self.params["idle_w"]), y=1, loc='left')
    plt.title("data_num" + str(self.params["data_num"]), loc='right')
    plt.title("delay_Weight" + str(self.params["delay_Weight"]))
    plt.ylabel('Reward')
    plt.xlabel('Time Frames')
    plt.legend()

    plt.show()

def plot_cost(self, isSmooth=False, window_size=5, polyorder=3):
    from more_itertools import chunked
    smoothed_list = [sum(x) for x in chunked(self.system_cost, self.T+1)]  ##一轮迭代的总奖励
    smoothed_top_list = [sum(x) for x in chunked(self.top_cost, self.T+1)]  ##一轮迭代的总奖励
    smoothed_close_cost = [sum(x) for x in chunked(self.close_cost, self.T+1)]  ##一轮迭代的总奖励
    smoothed_open_cost = [sum(x) for x in chunked(self.open_cost, self.T+1)]  ##一轮迭代的总奖励
    if isSmooth:
        # # 定义平滑参数，例如窗口大小和多项式阶数
        # window_size = 5  # 可根据实际情况调整
        # polyorder = 3  # 通常应小于窗口大小的一半

        # 应用平滑
        smoothed_list = savgol_filter(smoothed_list, window_size, polyorder)
        smoothed_top_list = savgol_filter(smoothed_top_list, window_size, polyorder)
        smoothed_close_cost = savgol_filter(smoothed_close_cost, window_size, polyorder)
        smoothed_open_cost = savgol_filter(smoothed_open_cost, window_size, polyorder)

    plt.plot(np.arange(len(smoothed_list)), smoothed_list, color='b', label='dqn')
    plt.plot(np.arange(len(smoothed_close_cost)), smoothed_close_cost, color='y', label='all_0')
    plt.plot(np.arange(len(smoothed_open_cost)), smoothed_open_cost, label='all_1')
    plt.plot(np.arange(len(smoothed_top_list)), smoothed_top_list, color='g', label='topk')
    plt.title("idle_w" + str(self.params["idle_w"]), y=1, loc='left')
    plt.title("data_num" + str(self.params["data_num"]), loc='right')
    plt.title("delay_Weight" + str(self.params["delay_Weight"]))
    plt.ylabel('Cost')
    plt.xlabel('Time Frames')
    plt.legend()
    plt.show()

def getData(self,name=None):
    from more_itertools import chunked
    smoothed_list = [sum(x) for x in chunked(self.system_cost, self.T + 1)]  ##一轮迭代的总奖励
    smoothed_top_list = [sum(x) for x in chunked(self.top_cost, self.T + 1)]  ##一轮迭代的总奖励
    smoothed_close_cost = [sum(x) for x in chunked(self.close_cost, self.T + 1)]  ##一轮迭代的总奖励
    smoothed_open_cost = [sum(x) for x in chunked(self.open_cost, self.T + 1)]
    difference_list = [b - a for a, b in zip(smoothed_list, smoothed_top_list)]
    # 对差值列表进行排序，并获取排序后的索引
    sorted_indices = sorted(range(len(difference_list)), key=lambda i: difference_list[i],reverse=True)
    num = 0
    minnum = 3000
    idx = sorted_indices[0]
    for i in range(len(sorted_indices)):
        if num >= 20:
            break
        if sorted_indices[i] > 1500:
            cost = [self.system_cost[sorted_indices[i] * 72:sorted_indices[i] * 72 + 72],
                    self.top_cost[sorted_indices[i] * 72:sorted_indices[i] * 72 + 72]]
            if (sum(cost[0] < sum(cost[1]))):
                count = 0
                num += 1
                for c, tc in zip(cost[0], cost[1]):
                    if tc < c:
                        count += 1
                if count < minnum:
                    minnum = count
                    idx = i
                    # print(sum(cost[0]))
                    # print(sum(cost[1]))



    energy = [[item[0] for item in self.cost_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
              [item[1] for item in self.cost_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
              [item[2] for item in self.cost_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
              self.top_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]]
    delay = [[item[0] for item in self.cost_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
             [item[1] for item in self.cost_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
             [item[2] for item in self.cost_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
             self.top_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]]
    cost = [self.system_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72],
            self.open_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72],
            self.close_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72],
            self.top_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72]]

    energy = [[item[0] for item in self.cost_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
              [item[1] for item in self.cost_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
              [item[2] for item in self.cost_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
              self.top_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]]
    delay = [[item[0] for item in self.cost_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
             [item[1] for item in self.cost_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
             [item[2] for item in self.cost_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
             self.top_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]]
    cost = [self.system_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72],
            self.open_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72],
            self.close_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72],
            self.top_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72]]
    action = [sum(item) for item in self.action[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72]]


    list = [sum(x) for x in chunked(self.reward_list, self.T + 1)]  ##一轮迭代的总奖励
    top_list = [sum(x) for x in chunked(self.rw_topk, self.T + 1)]
    close_cost = [sum(x) for x in chunked(self.rw_all_0, self.T + 1)]  ##一轮迭代的总奖励
    open_cost = [sum(x) for x in chunked(self.rw_all_1, self.T + 1)]  ##一轮迭代的总奖励
    if name is not None:
        np.savetxt(
            '实验结果/' + name + '.txt',[
                [delay,energy,cost,action],
                [smoothed_list,smoothed_open_cost,smoothed_close_cost,smoothed_top_list],
                [list,open_cost,close_cost,top_list]],
            delimiter=',', fmt='%s')
    return delay,energy,cost,action


def plot_ddd(map_list=None, ylab='Reward'):
    s = "ygbr"
    i = 0
    name = ['DQN-OD', '全部开启', '全部关闭', 'TOP']
    for mapi in map_list:
        plt.plot(np.arange(len(mapi)), mapi, color=s[i],label=name[i])
        i += 1
    plt.ylabel(ylab)
    plt.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.xlabel('Time Frames')
    plt.show()


def keep(my_object,name):
    with open(name, 'wb') as f:
        pickle.dump(my_object, f)
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def getDataWeight(self,name=None):
    from more_itertools import chunked
    smoothed_list = [sum(x) for x in chunked(self.system_cost, self.T + 1)]  ##一轮迭代的总奖励
    smoothed_top_list = [sum(x) for x in chunked(self.top_cost, self.T + 1)]  ##一轮迭代的总奖励
    smoothed_close_cost = [sum(x) for x in chunked(self.close_cost, self.T + 1)]  ##一轮迭代的总奖励
    smoothed_open_cost = [sum(x) for x in chunked(self.open_cost, self.T + 1)]
    difference_list = [b - a for a, b in zip(smoothed_list, smoothed_top_list)]
    # 对差值列表进行排序，并获取排序后的索引
    sorted_indices = sorted(range(len(difference_list)), key=lambda i: difference_list[i],reverse=True)

    idx = random.randint(0, 20)

    energy = [[item[0] for item in self.cost_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
              [item[1] for item in self.cost_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
              [item[2] for item in self.cost_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
              self.top_energy[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]]
    delay = [[item[0] for item in self.cost_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
             [item[1] for item in self.cost_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
             [item[2] for item in self.cost_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]],
             self.top_delay[sorted_indices[idx] * 720:sorted_indices[idx] * 720 + 720]]
    cost = [self.system_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72],
            self.open_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72],
            self.close_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72],
            self.top_cost[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72]]
    action = [sum(item) for item in self.action[sorted_indices[idx] * 72:sorted_indices[idx] * 72 + 72]]
    list = [sum(x) for x in chunked(self.reward_list, self.T + 1)]  ##一轮迭代的总奖励
    top_list = [sum(x) for x in chunked(self.rw_topk, self.T + 1)]
    close_cost = [sum(x) for x in chunked(self.rw_all_0, self.T + 1)]  ##一轮迭代的总奖励
    open_cost = [sum(x) for x in chunked(self.rw_all_1, self.T + 1)]  ##一轮迭代的总奖励
    if name is not None:
        np.savetxt(
            '实验结果/' + name + '.txt', [
                [delay, energy, cost, action],
                [smoothed_list, smoothed_open_cost, smoothed_close_cost, smoothed_top_list],
                [list, open_cost, close_cost, top_list]],
            delimiter=',', fmt='%s')
    return delay,energy,cost,action
