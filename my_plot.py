import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class My_plot:
    def __init__(self):
        self.maxmindelay = []
        self.maxminenergy = []

        self.loss_plot = []
        self.around_list = []
        self.load_num = []
        self.reward_list = []
        ## 与  n_time_step  opentime对应
        self.T = (18-6) * 6 - 1 ##决策数量 （24-6） * 6
        self.k = 60 / 6
        self.avg_delay = []

        self.rw_all_0 = []
        self.rw_all_1 = []

        ##[cost_delay, all_1_delay, all_0_delay]
        self.cost_delay = []

        self.cost_energy = []

        self.params = dict(
            {
               "delay_Weight": 0, ##environment
               "idea_w": 0, ##environment
               "data_num": 0,
            }
        )


    def plot_loss(self):
        plt.plot(np.arange(len(self.loss_plot)), self.loss_plot)
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


    def plot_reward(self):
        from more_itertools import chunked
        list = [sum(x) for x in chunked(self.reward_list, self.T)] ##一轮迭代的总奖励
        all_0_list = [sum(x) for x in chunked(self.rw_all_0, self.T)] ##一轮迭代的总奖励
        all_1_list = [sum(x) for x in chunked(self.rw_all_1, self.T)] ##一轮迭代的总奖励
        np.savetxt(
            'network_params/' + str(time.strftime('%H_%M_%S', time.localtime(time.time()))) + '_rw' + '.txt',
            list, fmt='%s')
        plt.plot(np.arange(len(list)), list, color='b')
        plt.plot(np.arange(len(all_0_list)), all_0_list, color='g')
        plt.plot(np.arange(len(all_1_list)), all_1_list, color='y')
        plt.title("idle_w" + str(self.params["idle_w"]), y=1, loc='left')
        plt.title("data_num" + str(self.params["data_num"]), loc='right')
        plt.title("delay_Weight" + str(self.params["delay_Weight"]))
        plt.ylabel('Reward')
        plt.xlabel('Time Frames')
        plt.show()

    def plot_default(self, map, map_list=None, ylab='Reward'):
        s = "ygb"
        i = 0
        if(map_list != None):
            map_list.append(map)
            for mapi in map_list:
                plt.plot(np.arange(len(mapi)), mapi, color=s[i])
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