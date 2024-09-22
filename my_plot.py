import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class My_plot:
    def __init__(self):
        self.param = []
        self.maxmindelay = []
        self.maxminenergy = []

        self.system_cost = []
        self.top_cost = []
        self.close_cost = []
        self.open_cost = []
        self.loss_plot = []
        self.around_list = []
        self.load_num = []
        self.reward_list = []
        self.rw_topk = []
        self.action = []
        ## 与  n_time_step  opentime对应
        self.T = (18-6) * 6 - 1 ##决策数量 （24-6） * 6
        self.k = 60 / 6
        self.avg_delay = []

        self.rw_all_0 = []
        self.rw_all_1 = []

        ##[cost_delay, all_1_delay, all_0_delay]
        self.cost_delay = []

        self.cost_energy = []

        self.top_delay = []
        self.top_energy = []
        self.params = dict(
            {
               "delay_Weight": 0, ##environment
               "idea_w": 0, ##environment
               "data_num": 0,
            }
        )


    def plot_loss(self):
        from more_itertools import chunked
        smoothed_list = [sum(x) for x in chunked(self.loss_plot, self.T + 1)]  ##一轮迭代的总奖励
        plt.plot(np.arange(len(smoothed_list)), smoothed_list)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

    def plot_avg_delay(self):
        from more_itertools import chunked
        list = [sum(x) for x in chunked(self.avg_delay, self.T)]
        plt.plot(np.arange(len(list)), list)
        plt.ylabel('AVG_DELAY')
        plt.xlabel('Time Frames')
        plt.show()

    def plot_load(self):
       ## plt.plot(pd.date_range(start='2007-07-20 06:00:00', periods=216, freq='5T'), self.load_num[:216])
       ## from more_itertools import chunked
        ##list = [sum(x) for x in chunked(self.around_list, 10)]
        ##plt.plot(np.arange(len(list)), list)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        num = (int)(self.T * self.k)
        plt.plot(np.arange(num), self.load_num[:num])
        plt.ylabel('负载')
        plt.xlabel('时隙')
        plt.savefig('load.svg', format='svg')
        plt.show()

    def plot_reward(self, isSmooth=False, window_size=5, polyorder=3, other=False, topk=False):
        from more_itertools import chunked
        smoothed_list = [sum(x) for x in chunked(self.reward_list, self.T + 1)]  ##一轮迭代的总奖励
        smoothed_top_list = [sum(x) for x in chunked(self.rw_topk, self.T + 1)]
        smoothed_close_cost = [sum(x) for x in chunked(self.rw_all_0, self.T + 1)]  ##一轮迭代的总奖励
        smoothed_open_cost = [sum(x) for x in chunked(self.rw_all_1, self.T + 1)]  ##一轮迭代的总奖励
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

    def plot_default(self, map, map_list=None, ylab='Reward'):
        s = "ygbr"
        name = ['DQN-OD', '全部开启', '全部关闭', 'TOP']
        i = 0
        if(map_list != None):
            map_list.append(map)
            for mapi in map_list:
                plt.plot(np.arange(len(mapi)), mapi, color=s[i],label=name[i])
                i += 1
        else:
            plt.plot(np.arange(len(map)), map)
        plt.title("idle_w" + str(self.params["idle_w"]), y=1, loc='left')
        plt.title("data_num" + str(self.params["data_num"]), loc='right')
        plt.title("delay_Weight" + str(self.params["delay_Weight"]))
        plt.ylabel(ylab)
        plt.xlabel('Time Frames')
        plt.show()
        ##env.my_plot.plot_default([item[0] for item in env.my_plot.cost_delay[-1079:]],[[item[1] for item in env.my_plot.cost_delay[-1079:]],[item[2] for item in env.my_plot.cost_delay[-1079:]]])

    def plot_cost(self, isSmooth=False, window_size=5, polyorder=3):
        from more_itertools import chunked
        smoothed_list = [sum(x) for x in chunked(self.system_cost, self.T + 1)]  ##一轮迭代的总奖励
        smoothed_top_list = [sum(x) for x in chunked(self.top_cost, self.T + 1)]  ##一轮迭代的总奖励
        smoothed_close_cost = [sum(x) for x in chunked(self.close_cost, self.T + 1)]  ##一轮迭代的总奖励
        smoothed_open_cost = [sum(x) for x in chunked(self.open_cost, self.T + 1)]  ##一轮迭代的总奖励
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


    def getData(self):
        from more_itertools import chunked
        smoothed_list = [sum(x) for x in chunked(self.system_cost, self.T + 1)]  ##一轮迭代的总奖励
        smoothed_top_list = [sum(x) for x in chunked(self.top_cost, self.T + 1)]  ##一轮迭代的总奖励

        difference_list = [b - a for a, b in zip(smoothed_list, smoothed_top_list)]
        # 对差值列表进行排序，并获取排序后的索引
        sorted_indices = sorted(range(len(difference_list)), key=lambda i: difference_list[i])
        energy = [[item[0] for item in self.cost_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
                  [item[1] for item in self.cost_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
                  [item[2] for item in self.cost_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
                  self.top_energy[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]]
        delay = [[item[0] for item in self.cost_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
                 [item[1] for item in self.cost_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
                 [item[2] for item in self.cost_delay[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]],
                 self.top_cost[sorted_indices[0] * 720:sorted_indices[0] * 720 + 720]]

        cost = [self.system_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72],
                self.open_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72],
                self.close_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72],
                self.top_cost[sorted_indices[0] * 72:sorted_indices[0] * 72 + 72]]

        return delay,energy,cost