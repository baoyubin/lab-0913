import numpy as np
import transbigdata as tbd
from my_env import Env
from my_map import Bus_map, Load_map
from my_plot import My_plot
import math
import copy
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
import time
import torch
import torch.nn as nn



class greedy_env(Env):

    def step(self, action, avg_list):
        self.obs_taxi = self._get_W()
        self.obs_bus = self._get_P()

        ##处理负载计算
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))
        ##W_nonzero_index = W_one.nonzero()
        self.my_plot.load_num.append(sum(W_one))
        sum_load = sum(W_one)
        cost_offalltoecd, tran_allenergy = self._get_cost_ecd(sum_load)
        obs_offloadtoBus = 0
        self.cost_place = 0
        avg_load = 0
        for index in np.arange(10):
            around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, False)
            around_load = sum(around.flatten())
            avg_load += around_load
        avg_load /= 10
        avg_list.append(avg_load)
        for index in np.arange(10):
            around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, False)
            around_load = sum(around.flatten())
            delay = self._get_cost_bus(around_load)
            delay_loc, energy_loc = self._get_cost_ecd(sum_load-around_load)
            ##if(delay * around_load + delay_loc * (sum_load-around_load) < cost_offalltoecd * sum_load):
            if (around_load >= np.mean(avg_list)):
                action[self.T][index] = 1
                obs_offloadtoBus = obs_offloadtoBus + delay * around_load / sum_load
                not_use = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
            self.my_plot.around_list.append(around_load)

        W_load = sum(W_one.flatten())
        obs_offloadtoEcd, tran_loadenergy = self._get_cost_ecd(W_load)
        obs_offloadtoEcd *= W_load / sum_load
        reward = self._get_reward(obs_offloadtoEcd, obs_offloadtoBus, cost_offalltoecd)  # TODO
        self.my_plot.reward_list.append(reward)

        self.observation_space = self._get_obs()
        done = self._get_done()  # TODO
        info = None  # TODO
        ##print(self.cost_place)
        self.obs_action = action[self.T]
        self.T += 1
        return copy.deepcopy(self.observation_space), reward, done, info

class topk_env(Env):

    def getTopK(self, p):
        self.obs_taxi = self._get_W()
        self.obs_bus = self._get_P()
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))
        around_list = []
        for index in np.arange(10):
            around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, False)
            around_load = sum(around.flatten())
            around_list.append(around_load)
        # print('around_list',  around_list)
        sorted_id = sorted(range(len(around_list)), key=lambda i:  around_list[i], reverse=True)
        # print('元素索引序列：', sorted_id)
        action = np.zeros(shape=10, dtype=np.int32)
        # for i in sorted_id:
        #     if k > 0:
        #         action[i] = 1
        #         k -= 1
        for index, value in enumerate(around_list):
            if value >= p:
                action[index] = 1
        return action



class topk2_env(Env):

    def getTopK(self, p):
        self.obs_taxi = self._get_W()
        self.obs_bus = self._get_P()
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))
        around_list = []
        for index in np.arange(10):
            around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
            around_load = sum(around.flatten())
            around_list.append(around_load)
        # print('around_list',  around_list)
        sorted_id = sorted(range(len(around_list)), key=lambda i:  around_list[i], reverse=True)
        # print('元素索引序列：', sorted_id)
        action = np.zeros(shape=10, dtype=np.int32)
        # for i in sorted_id:
        #     if k > 0:
        #         action[i] = 1
        #         k -= 1
        for index, value in enumerate(around_list):
            if value >= p:
                action[index] = 1
        return action

class kmeans_env(Env):
    def getLocation(self, obs_bus):
        location = []
        for i in obs_bus:
            x = math.ceil(i / 45)
            y = i % 45
            location.append([x, y])
        return location
    def getAction(self, points, given_points):
        # 生成10个点的坐标 生成5个给定点的坐标
        # 计算10个点到5个给定点的欧式距离
        distances = np.linalg.norm(points[:, np.newaxis] - given_points, axis=2)
        # 对于每个给定点，找到距离最近的点的索引
        min_distances_index = np.argmin(distances, axis=0)
        # 选择对应的点
        #min_distances_points = points[min_distances_index]
        #print(min_distances_points)
        action = np.zeros(shape=10, dtype=np.int32)
        for i in min_distances_index:
            action[i] = 1
        return action



    def getTopK(self, k):
        self.obs_taxi = self._get_W()
        self.obs_bus = self._get_P()
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))
        around_list = []

        array = []

        for i, x in np.ndenumerate(W_two):
            #print(i, x)
            if x > 0:
                array.append(i)

        kmeans = KMeans(n_clusters=k).fit(array)

        action = self.getAction(np.array(self.getLocation(self.obs_bus)), kmeans.cluster_centers_)

        return action



if __name__ == '__main__':

    mode = 'topk'
    my_plot = My_plot()
    bounds = [121.4, 31.15, 121.6, 31.35]
    params = tbd.area_to_params(bounds, accuracy=500, method='rect')
    k = 6
    taxi_map = Load_map()
    taxi_map.get_map(bounds, params)
    bus_map = Bus_map()
    bus_map.get_map(bounds, params)
    map_x = math.ceil((bounds[2] - bounds[0]) / params['deltalon'])
    map_y = math.ceil((bounds[3] - bounds[1]) / params['deltalat'])

    if mode == 'greedy':
       env = greedy_env(map_x, map_y, taxi_map.load_map, bus_map.bus_map, my_plot)
       s = env.reset()
       action = np.zeros(shape=(len(env.load_map), env.bus_num), dtype=np.int32)
       n_episode = 1
       n_time_step = 72
       avg_list = []
       for episode_i in range(n_episode):
           episode_reward = 0
           for step_i in range(n_time_step):
               s_, r, done, info = env.step(action, avg_list)
               episode_reward += r
               if done:
                   s = env.reset()
                   break
       plt.plot(np.arange(len(my_plot.reward_list)), my_plot.reward_list)
       plt.ylabel('Reward')
       plt.xlabel('Time Frames')
       plt.show()
       print(episode_reward)
    if (mode == 'topk'):
        env = topk_env(map_x, map_y, taxi_map.load_map, bus_map.bus_map, my_plot)
        p = env.bus_max*2/10;
        s = env.reset()
        #action = np.zeros(shape=(len(env.load_map), env.bus_num), dtype=np.int32)
        n_episode = 1
        n_time_step = 72
        avg_list = []
        aaa = np.empty(n_time_step)
        for episode_i in range(n_episode):
            episode_reward = 0
            for step_i in range(n_time_step):
                action = env.getTopK(p)
                aaa[step_i] = sum(action)
                s_, r, done, info = env.step(action)
                episode_reward += r
                if done:
                    s = env.reset()
                    break
        my_plot.params["delay_Weight"] = env.delay_Weight
        my_plot.params["data_num"] = env.data_num
        my_plot.params["idle_w"] = env.idle_w
        my_plot.plot_reward()
        energy = [[item[0] for item in env.my_plot.cost_energy[-1080:]],
                  [item[1] for item in env.my_plot.cost_energy[-1080:]],
                  [item[2] for item in env.my_plot.cost_energy[-1080:]]]
        delay = [[item[0] for item in env.my_plot.cost_delay[-1080:]],
                 [item[1] for item in env.my_plot.cost_delay[-1080:]],
                 [item[2] for item in env.my_plot.cost_delay[-1080:]]]

        env.my_plot.plot_default(delay[0], [delay[1], delay[2]], "delay")
        ##[cost_delay, all_1_delay, all_0_delay]
        env.my_plot.plot_default(energy[0], [energy[1], energy[2]], "energy")

        c = [(delay[0][i] - delay[1][i]) / delay[1][i] for i in range(0, len(delay[0]))]
        d = [(energy[1][i] - energy[0][i]) / energy[1][i] for i in range(0, len(energy[0]))]

        print("END")
        print(episode_reward)
        print(env.delay_Weight, env.data_num, env.idle_w)
        print("==============")
        print("dealy:", sum(c) / len(c), "energy", sum(d) / len(d))

    if (mode == 'kmeans'):
        env = kmeans_env(map_x, map_y, taxi_map.load_map, bus_map.bus_map, my_plot)
        s = env.reset()
        #action = np.zeros(shape=(len(env.load_map), env.bus_num), dtype=np.int32)
        n_episode = 1
        n_time_step = 72
        avg_list = []

        for episode_i in range(n_episode):
            episode_reward = 0
            for step_i in range(n_time_step):
                action = env.getTopK(k)
                s_, r, done, info = env.step(action)
                episode_reward += r
                if done:
                    s = env.reset()
                    break
        my_plot.params["delay_Weight"] = env.delay_Weight
        my_plot.params["data_num"] = env.data_num
        my_plot.params["idle_w"] = env.idle_w
        my_plot.plot_reward()
        energy = [[item[0] for item in env.my_plot.cost_energy[-1080:]],
                  [item[1] for item in env.my_plot.cost_energy[-1080:]],
                  [item[2] for item in env.my_plot.cost_energy[-1080:]]]
        delay = [[item[0] for item in env.my_plot.cost_delay[-1080:]],
                 [item[1] for item in env.my_plot.cost_delay[-1080:]],
                 [item[2] for item in env.my_plot.cost_delay[-1080:]]]

        env.my_plot.plot_default(delay[0], [delay[1], delay[2]], "delay")
        ##[cost_delay, all_1_delay, all_0_delay]
        env.my_plot.plot_default(energy[0], [energy[1], energy[2]], "energy")

        c = [(delay[0][i] - delay[1][i]) / delay[1][i] for i in range(0, len(delay[0]))]
        d = [(energy[1][i] - energy[0][i]) / energy[1][i] for i in range(0, len(energy[0]))]

        print("END")
        print(episode_reward)
        print(env.delay_Weight, env.data_num, env.idle_w)
        print("==============")
        print("dealy:", sum(c) / len(c), "energy", sum(d) / len(d))

    lllll = str(time.strftime('%H_%M_%S', time.localtime(time.time())))
    if mode == "kmeans":
        np.savetxt('kmeans/' + lllll + '_' + str(env.delay_Weight) + '_' + str(n_episode) + '_' + str(
            env.data_num) + '_' + str(env.B_ecd / env.B_bus)
                   + '_' + str(env.ECD_CPU_frequency / env.BUS_CPU_frequency) + '_' + str(env.bus_bound) + '.txt',
                   [env.my_plot.reward_list[-107:], env.my_plot.rw_all_1[-107:], env.my_plot.rw_all_0[-107:], energy,
                    delay], fmt='%s')
    elif mode == 'topk':
        np.savetxt('topk/' + str(env.busidle)+ '_' + str(max(max(delay))) + '_' + str(p) + '_' + str(env.delay_Weight) + '_' + str(n_episode) + '_' + str(
            env.data_num) + '_' + str(env.B_ecd)
                   + '_' + str(env.ECD_CPU_frequency) + '_' + str(env.bus_bound) + '.txt',
                   [env.my_plot.reward_list[-n_time_step:], env.my_plot.rw_all_1[-n_time_step:], env.my_plot.rw_all_0[-n_time_step:], energy,
                    delay], fmt='%s')
        np.savetxt(
            'topk/' + str(max(max(delay))) + '_' + lllll + '_' + '.txt',
            [aaa], delimiter=',',fmt='%s')
        np.savetxt(
            'topk/' + str(max(max(delay))) + '_delay' + lllll + '_' + '.txt',
            [env.my_plot.maxmindelay], delimiter=',', fmt='%s')
        np.savetxt(
            'topk/' + str(max(max(delay))) + '_energy' + lllll + '_' + '.txt',
            [env.my_plot.maxminenergy], delimiter=',', fmt='%s')