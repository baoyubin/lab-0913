import numpy as np
import copy
import math
from my_cost import Cost
class Env:
    def __init__(self, x, y, load_map, bus_map, my_plot, delay_Weight=0.5, idle_w=0.5, data_num=1):
        self.cover =[]

        self.eswitch = 0
        self.ecdidle = 30
        self.busidle = 10
        self.opentime = 10

        self.x = x
        self.y = y
        self.T = 0
        self.cost_func = Cost()
        # self.ae = 0.001
        # self.at = 10

        # self.ae = 0.003
        # self.at = 1
        self.g_bus = 4.11 * (3 * 10 ** 2 / (4 * math.pi * 915 * 0.75)) ** (2.8)
        self.g_mec = 4.11 * (3 * 10 ** 2 / (4 * math.pi * 915 * 2.5)) ** (2.8)
        self.maxload = 0


        # self.maxdelay = 1.5
        # self.mindelay = 0
        # self.maxenergy = 700
        # self.minenergy = 200 #论文结果
        self.ae = 0.001
        self.at = 5
        self.maxdelay = 1.5 #1.5
        self.mindelay = 0
        self.maxenergy = 400 #400
        self.minenergy = 140 #论文结果2

        # self.maxdelay = 1.5
        # self.mindelay = 0
        # self.maxenergy = 500
        # self.minenergy = 140  # 论文结果2 对比能耗实验

        ##plot
        self.my_plot = my_plot
        self.rw_local = 0

        #busbound=5
        self.bus_num = 10
        self.action_num = 10
        self.bus_bound = 6

        self.bus_map = bus_map
        self.load_map = load_map

        self.delay_Weight = delay_Weight
        self.energy_Weight = 1 - self.delay_Weight

        self.idle_w = idle_w


        ## HZ 10 * 10**6 原来的 20mhz 20mhz
        self.B_bus = 20 * 10**6
        self.B_ecd = 20 * 10**6
        self.B_cloud = self.B_ecd

        ## HZ 5*10**9 原来的 10 30
        self.BUS_CPU_frequency = 10 * 10**9
        self.ECD_CPU_frequency = 20 * 10**9
        self.Cloud_CPU_frequency = 20 * 10**9

        self.cost_place = 0

        ##queue TODO
        self.queue = []
        self.T_FINISH = 500
        self.data_num = data_num
        self.service_rate = 100
        self.bus_max = 100 ## 真实队列为*datanum
        self.P_bus = 25 ##W
        self.P_ecd = 50
        self.observation_space = dict(
            {
               "taxi": np.zeros(shape=(self.x*self.y), dtype=np.int32), ##environment
               "bus": np.zeros(shape=self.bus_num, dtype=np.int32), ##environment
               "bus_action": np.zeros(shape=self.bus_num, dtype=np.uint8),
            }
        )

        ##self.action_space = np.random.randint(0, 2, self.bus_num) ##action[0 1]
        self.action_space = np.zeros(shape=self.bus_num, dtype=np.int32)
        self.obs_taxi = self.observation_space["taxi"]
        self.obs_bus = self.observation_space["bus"]
        self.obs_action = self.observation_space["bus_action"]
    def reset(self):
        self.T = 0
        self.obs_taxi = self._get_W()
        self.obs_bus = self._get_P()
        ##self.T = 1
        self.obs_action = self.action_space
        self.observation_space = self._get_obs()
        return copy.deepcopy(self.observation_space)

    def _get_obs(self):
        return {"taxi": self.obs_taxi, "bus": self.obs_bus, "bus_action": self.obs_action}

    def _get_cost_bus2(self, off_sum):  # TODO
        assert off_sum <= self.bus_max
        idle = 0
        cost = 40
        if (off_sum == 0):

            return 0, 0 + cost
        off_sum *= self.data_num
        ## 每个任务的计算时间 平均延迟
        computingDelay_Bus = self.cost_func.get_computation_delay(off_sum, self.BUS_CPU_frequency)
        assert (computingDelay_Bus <= 120)
        # ## 总能耗 ：不需要*任务数量
        # exe_energy = computingDelay_Bus * self.P_bus

        # ## 每个任务的传输延迟
        cost_trantoBus = self.cost_func.get_transmission_delay(self.B_bus, off_sum, self.g_bus)
        # ## 总传输能耗
        # tran_energy = cost_trantoBus * self.cost_func.P_t * off_sum

        return cost_trantoBus + computingDelay_Bus,  cost + 0.5*cost*off_sum/self.bus_max
    def _get_cost_bus(self, off_sum): #TODO
        # return self._get_cost_bus(off_sum)
        assert off_sum <= self.bus_max
        idle = 0
        cost = self.busidle #30
        if(off_sum == 0):
            # cost = (self.P_bus * self.idle_w + (self.P_bus - self.P_bus * self.idle_w) * 1 / (
            #         self.bus_max)) * 60
            return 0, 0 + cost
        off_sum *= self.data_num
        ## 每个任务的计算时间 平均延迟
        computingDelay_Bus = self.cost_func.get_computation_delay(off_sum, self.BUS_CPU_frequency)
        assert (computingDelay_Bus <= 120)
        ## 总能耗 ：不需要*任务数量
        exe_energy = computingDelay_Bus * self.P_bus

        ## 每个任务的传输延迟
        cost_trantoBus = self.cost_func.get_transmission_delay(self.B_bus, off_sum, self.g_bus)
        ## 总传输能耗
        tran_energy = cost_trantoBus * self.cost_func.P_t * off_sum



        return cost_trantoBus + computingDelay_Bus, tran_energy + cost + exe_energy

    def cloud_get_ecd(self, load_sum):
        allsum = load_sum * self.data_num
        cost = self.ecdidle #50
        if (load_sum == 0):
            return 0, 0 + cost
        cost_queue = 0
        load_sum *= self.data_num
        ## cost_trantoEcd为平均传输时间
        cost_trantoEcd = self.cost_func.get_transmission_delay(self.B_ecd, allsum, self.g_mec)
        ## 总传输时间 = 卸载任务数量 * 每个任务传输延迟 cost_trantoEcd * load_sum
        ## 总传输能耗 tran_energy = cost_trantoEcd * self.cost_func.P_t * load_sum
        tran_energy = cost_trantoEcd * self.cost_func.P_t * load_sum
        ecd_size = allsum
        ## 云
        tran_cloud = 0
        exe_cloud = 0
        cloud_size = 0
        if (load_sum > self.service_rate):
            # 2 * self.service_rate - load_sum > 0
            if 2 * self.service_rate - load_sum > 0:
                cost_queue = (load_sum - self.service_rate) / (
                            2 * self.service_rate * (2 * self.service_rate - load_sum))
            else:
                ecd_size = 2 * self.service_rate - 1
                cost_queue = (ecd_size - self.service_rate) / (
                        2 * self.service_rate * (2 * self.service_rate - ecd_size))
                cloud_size = allsum - ecd_size
                while (cloud_size > 0):
                    if cloud_size > 2 * self.service_rate - 1:
                        ecd_size = 2 * self.service_rate - 1
                        cost_queue += (ecd_size - self.service_rate) / (
                                2 * self.service_rate * (2 * self.service_rate - ecd_size))
                        cloud_size -= ecd_size
                    else:
                        cost_queue += (cloud_size - self.service_rate) / (
                                2 * self.service_rate * (2 * self.service_rate - cloud_size))
                        cloud_size = 0
                ## todo 修改模型
                # tran_cloud = self.cost_func.get_transmission_delay(self.B_cloud, cloud_size)
                # if cloud_size > 2 * self.service_rate:
                #     tran_cloud = 3
                # elif cloud_size > self.service_rate:
                #     tran_cloud = 2
                # else:
                #     tran_cloud = 1.5
                # tran_cloud = 2 * ( - self.service_rate)/self.service_rate
                # exe_cloud = self.cost_func.get_computation_delay(cloud_size, self.Cloud_CPU_frequency)
        ## 不考虑云，及排队时延不会超过2*服务率 2.29 145
        computingDelay_ECD, exe_energy = self.ecd_not_cloud(allsum)
        #### 考虑云
        # computingDelay_ECD, exe_energy = self.ecd_not_cloud(allsum - cloud_size)
        ### 调试用
        # tt = (tran_cloud * cloud_size + computingDelay_ECD * ecd_size + exe_cloud * cloud_size) / allsum
        # aa, ee = self.ecd_not_cloud(allsum)
        # assert aa < tt
        ### 修改的
        # delay = (cost_trantoEcd * allsum + tran_cloud * cloud_size + computingDelay_ECD * ecd_size + exe_cloud * cloud_size) / allsum
        delay = computingDelay_ECD + cost_trantoEcd
        # cost_queue = (load_sum / self.service_rate * (self.service_rate - load_sum)) / 100
        # return computingDelay_ECD + cost_trantoEcd + cost_queue, tran_energy
        return delay + cost_queue, tran_energy + exe_energy + cost

    def cloud_get_ecd2(self, load_sum):
        allsum = load_sum * self.data_num
        cost = 238
        if (load_sum == 0):
            return 0, 0 + cost
        ## cost_trantoEcd为平均传输时间
        cost_trantoEcd = self.cost_func.get_transmission_delay(self.B_ecd, allsum, self.g_mec)
        ## 总传输时间 = 卸载任务数量 * 每个任务传输延迟 cost_trantoEcd * load_sum
        ## 总传输能耗 tran_energy = cost_trantoEcd * self.cost_func.P_t * load_sum
        tran_energy = cost_trantoEcd * self.cost_func.P_t * load_sum
        ## 云
        tran_cloud = 0
        cloud_size = 0
        ecd_size = allsum
        if (load_sum > self.service_rate):
                cloud_size = allsum - self.service_rate
                tran_cloud = 3 * (cloud_size) / self.service_rate
                ecd_size = self.service_rate
        computingDelay_ECD, exe_energy = self.ecd_not_cloud(ecd_size)

        ### 调试用
        tt = (tran_cloud * cloud_size + computingDelay_ECD * ecd_size) / (allsum)
        aa, ee = self.ecd_not_cloud(allsum)
        # assert (cloud_size+ecd_size) == allsum
        # if (aa > tt):
        #     print(aa)
        #     print(tt)
        #     print(allsum)
        #     print(cloud_size)
        # assert aa <= tt
        ### 修改的
        delay = (cost_trantoEcd * allsum + tran_cloud * cloud_size + computingDelay_ECD * ecd_size) / allsum
        # delay = computingDelay_ECD + cost_trantoEcd
        # cost_queue = (load_sum / self.service_rate * (self.service_rate - load_sum)) / 100
        # return computingDelay_ECD + cost_trantoEcd + cost_queue, tran_energy
        return delay, (cost+0.5*cost*ecd_size/self.service_rate+0.1*cost*cloud_size/self.service_rate)
    def ecd_not_cloud(self, allsum):
        ## (平均计算延迟)计算延迟 =  每个任务计算延迟
        computingDelay_ECD = self.cost_func.get_computation_delay(allsum, self.ECD_CPU_frequency)
        exe_energy = computingDelay_ECD * self.P_ecd
        return computingDelay_ECD, exe_energy
    def _get_cost_ecd(self, load_sum):


        allsum = load_sum * self.data_num
        load_sum *= self.data_num
        if(load_sum == 0):
            return 0, 0

        cost_trantoEcd = self.cost_func.get_transmission_delay(self.B_ecd, allsum, self.g_mec)
        tran_cloud = 0
        cloud_size = 0

        if (load_sum > self.service_rate):
            cloud_size = allsum - self.service_rate
            tran_cloud = 1 * (cloud_size) / allsum
        delay = (cost_trantoEcd * allsum + tran_cloud * cloud_size) / allsum
        return delay, tran_energy

    def _get_reward_s(self, cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy, all_1_energy,
                    action):  # TODO
        self.my_plot.avg_delay.append(cost_delay)

        delay = (cost_delay - all_1_delay) / (all_0_delay - all_1_delay)
        cost = cost_energy / 3000
        assert cost < 1
        rew = (1 - self.delay_Weight) * cost /self.delay_Weight * delay
        ra = (1 - self.delay_Weight) * cost - self.delay_Weight * delay
        return ra


    def _get_reward(self, cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy, all_1_energy, action): #TODO
        self.my_plot.avg_delay.append(cost_delay)
        ##return (cost_offalltoecd - obs_offloadtoBus - obs_offloadtoEcd) / cost_offalltoecd - self.rw_local / 10 * 0.3
        ##return (cost_offalltoecd - obs_offloadtoBus - obs_offloadtoEcd)
        #return -0.5*(obs_offloadtoBus+obs_offloadtoEcd)-0.1*self.cost_place

        # if cost_delay > all_0_delay or cost_energy > all_1_energy:
        #     return -100

        # if cost_delay < all_1_delay and cost_delay < all_0_delay and cost_energy < all_1_energy:
        #     return 0

        delay = (cost_delay - all_1_delay) / (all_0_delay - all_1_delay)
        energy = (cost_energy - all_0_energy) / (all_1_energy - all_0_energy)

        min_delay = min(all_1_delay, all_0_delay, cost_delay)
        min_energy = min(cost_energy, all_0_energy, all_1_energy)
        max_energy = max(cost_energy, all_0_energy, all_1_energy)
        max_delay = max(all_1_delay, all_0_delay, cost_delay)
        delay = (cost_delay - min_delay) / (max_delay - min_delay)
        energy = (cost_energy - min_energy) / (max_energy - min_energy)
        # delay = cost_delay
        # energy = cost_energy / 100
        ##assert delay > 0
        ##assert energy > 0
        ##delay = math.log(cost_delay, 10) / math.log(all_0_delay, 10)
        ##energy = math.log(cost_energy, 10) / math.log(all_1_energy, 10)

        # delay = (cost_delay - all_1_delay) / (all_0_delay - all_1_delay)
        # energy = (cost_energy - all_0_energy) / (all_1_energy - all_0_energy)


        # rew = - self.delay_Weight * delay - self.energy_Weight * energy

        # rew = - self.delay_Weight * cost_delay * self.at - self.energy_Weight * cost_energy * self.ae

        delay = (cost_delay - self.mindelay) / (self.maxdelay - self.mindelay)
        energy = (cost_energy - self.minenergy) / (self.maxenergy - self.minenergy)
        rew = (- self.delay_Weight * delay - self.energy_Weight * energy)
        # rew = delay/energy
        # r0 = - self.delay_Weight * all_0_delay * self.at - self.energy_Weight * all_0_energy * self.ae
        # r1 = - self.delay_Weight * all_1_delay * self.at - self.energy_Weight * all_1_energy * self.ae
        ## todo 改成 min=0 max=x，后续根据值进行更新最大值
        # if rew > 0:
        #     return 0
        ## assert rew > 0
        self.my_plot.maxmindelay.append(delay)
        self.my_plot.maxminenergy.append(energy)
        return rew

    ##all_1 不一定是最小延迟
    def sigmoid(self, X, useStatus):
        if useStatus:
            return 1.0 / (1 + np.exp(-float(X)));
        else:
            return float(X)

    def _get_done(self): #TODO
        a = len(self.load_map)
        if self.T == a-1:
            return 1
        return 0

    def step(self, action):#TODO
        self.rw_local = 0
        rw_all_0 = 0
        rw_all_1 = 0
        opentime = self.opentime
        naction = self.obs_action
        num = 0
        for i in range(10):
            if naction[i] != action[i]:
                num += 1
        for t in range(opentime):
            ##read the data
            self.obs_taxi = self._get_W()
            self.obs_bus = self._get_P()
            self.obs_action = action

            ##处理负载计算
            W_one = copy.deepcopy(self.obs_taxi)
            W_two = W_one.reshape((self.y, -1))
            ##W_nonzero_index = W_one.nonzero()
            ##sss = sum(W_one)
            s1 = W_two[0:22, 0:18]
            s2 = W_two[22:, 0:18]
            s3 = W_two[0:22:, 18:]
            s4 = W_two[22::, 18:]
            ##sum(map(sum, s1))
            self.my_plot.load_num.append(sum(W_one))
            sum_load = sum(W_one)
            ## 无用
            # n_all_0_delay, n_all_0_energy = self._get_cost_ecd(sum_load)

            ## 延迟为每个任务的平均延迟，能耗为总能耗
            s1_0_delay, s1_0_energy = self._get_cost_ecd(sum(sum(s1)))
            s2_0_delay, s2_0_energy = self._get_cost_ecd(sum(sum(s2)))
            s3_0_delay, s3_0_energy = self._get_cost_ecd(sum(sum(s3)))
            s4_0_delay, s4_0_energy = self._get_cost_ecd(sum(sum(s4)))
            all_0_delay = (s1_0_delay * sum(sum(s1)) + s2_0_delay * sum(sum(s2)) + s3_0_delay * sum(
                sum(s3)) + s4_0_delay * sum(sum(s4))) / sum(W_one)
            all_0_energy = (s1_0_energy + s4_0_energy + s3_0_energy + s2_0_energy)
            all_1_delay, all_1_energy = self.get_bus_allopen(sum_load)

            Bus_delay = 0
            Bus_energy = 0

            self.cost_place = 0
            for index, act in enumerate(action):  # TODO
                around_load = 0
                if act:
                    around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
                    around_load = sum(around.flatten())
                    ###
                    if (around_load > self.bus_max):  #TODO
                        t = around_load - self.bus_max
                        bindex = self.obs_bus[index]
                        x = int(bindex / self.x)
                        y = bindex % self.x
                        W_two[x, y] = t
                        around_load = self.bus_max
                    ###
                    delay, energy = self._get_cost_bus(around_load)
                    Bus_delay += delay * around_load / sum_load
                    Bus_energy += energy
                self.my_plot.around_list.append(around_load)
            W_load = sum(W_one.flatten())

            s1 = W_two[0:22, 0:18]
            s2 = W_two[22:, 0:18]
            s3 = W_two[0:22:, 18:]
            s4 = W_two[22::, 18:]
            s1_0_delay, s1_0_energy = self._get_cost_ecd(sum(sum(s1)))
            s2_0_delay, s2_0_energy = self._get_cost_ecd(sum(sum(s2)))
            s3_0_delay, s3_0_energy = self._get_cost_ecd(sum(sum(s3)))
            s4_0_delay, s4_0_energy = self._get_cost_ecd(sum(sum(s4)))
            dqn_delay = (s1_0_delay * sum(sum(s1)) + s2_0_delay * sum(sum(s2)) + s3_0_delay * sum(
                sum(s3)) + s4_0_delay * sum(sum(s4))) / sum(W_one)
            dqn_energy = (s1_0_energy + s4_0_energy + s3_0_energy + s2_0_energy)
            ## 无用
            ## dqn_delay, dqn_energy = self._get_cost_ecd(W_load)
            ## dqn_delay *= W_load / sum_load

            cost_delay = dqn_delay + Bus_delay
            cost_energy = dqn_energy + Bus_energy
            if (t==0):
                cost_energy += num * self.eswitch
            reward = self._get_reward(cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy,
                                      all_1_energy, action)  # TODO
            # reward = self._get_reward_s(cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy,
            #                           all_1_energy, action)
            self.my_plot.cost_delay.append([cost_delay, all_1_delay, all_0_delay])
            self.my_plot.cost_energy.append([cost_energy, all_1_energy, all_0_energy])

            #3- self.delay_Weight * delay - self.energy_Weight * energy
            self.rw_local += reward

            # rw_all_0 += -self.delay_Weight * (all_0_delay - all_1_delay) / (all_0_delay - all_1_delay)
            # rw_all_1 += -self.energy_Weight * (all_1_energy - all_0_energy) / (all_1_energy - all_0_energy)
            # rw_all_0 += - self.delay_Weight * all_0_delay * self.at - self.energy_Weight * all_0_energy * self.ae
            # rw_all_1 += - self.delay_Weight * all_1_delay * self.at - self.energy_Weight * all_1_energy * self.ae
            rw_all_0 += - self.delay_Weight * (all_0_delay - self.mindelay) / (self.maxdelay - self.mindelay) - self.energy_Weight * (all_0_energy - self.minenergy) / (self.maxenergy - self.minenergy)
            rw_all_1 += - self.delay_Weight * (all_1_delay - self.mindelay) / (self.maxdelay - self.mindelay) - self.energy_Weight * (all_1_energy - self.minenergy) / (self.maxenergy - self.minenergy)

            # assert (all_1_energy >= all_0_energy)
            ##assert (all_1_energy >= cost_energy)
            ##assert (cost_delay <= all_0_delay)
            # assert (all_1_delay <= all_0_delay)
            assert (cost_delay <= self.maxdelay and all_0_delay <= self.maxdelay)
            assert (cost_delay >= self.mindelay and all_0_delay >= self.mindelay)
            assert (cost_energy <= self.maxenergy and all_1_energy <= self.maxenergy)
            assert (cost_energy >= self.minenergy and all_0_energy >= self.minenergy)

            self.observation_space = self._get_obs()
            done = self._get_done()  # TODO
            info = None  # TODO
            if done:
                return copy.deepcopy(self.observation_space), self.rw_local, done, info
            ##print(self.cost_place)
            self.T += 1

        # self.my_plot.rw_all_0.append(rw_all_0 / opentime)
        # self.my_plot.rw_all_1.append(rw_all_1 / opentime)
        # self.my_plot.reward_list.append(self.rw_local / opentime)
        # return copy.deepcopy(self.observation_space), self.rw_local / opentime, done, info
        self.my_plot.rw_all_0.append(rw_all_0)
        self.my_plot.rw_all_1.append(rw_all_1)
        self.my_plot.reward_list.append(self.rw_local)
        return copy.deepcopy(self.observation_space), self.rw_local, done, info

    def action_space_sample(self):
        return np.random.randint(0, 2, self.action_num)

    def get_around(self, index, W_two, bound, mode):
        x = int(index / self.x)
        y = index % self.x
        ##print(x, y)
        bound_left_x = x - bound if (x - bound > 0) else 0
        bound_left_y = y - bound if (y - bound > 0) else 0
        bound_right_x = x + bound + 1
        bound_right_y = y + bound + 1
        around = copy.deepcopy(W_two[bound_left_x:bound_right_x, bound_left_y:bound_right_y])
        if mode == True:
            W_two[bound_left_x:bound_right_x, bound_left_y:bound_right_y] = 0
        return around

    # def get_around(self, index, W_two):
    #     x = int(index / self.x)
    #     y = index % self.x
    #     if y == 0 and x == 0:
    #         around = copy.deepcopy(W_two[x:(x + 2), y:(y + 2)])
    #         W_two[x:(x + 2), y:(y + 2)] = 0
    #         return around
    #     elif y == 0:
    #         around = copy.deepcopy(W_two[(x - 1):(x + 2), y:(y + 2)])
    #         W_two[(x - 1):(x + 2), y:(y + 2)] = 0
    #         return around
    #     elif x == 0: ##右边越界自动处理
    #         around = copy.deepcopy(W_two[x:(x + 2), (y - 1):(y + 2)])
    #         W_two[x:(x + 2), (y - 1):(y + 2)] = 0
    #         return around
    #     around = copy.deepcopy(W_two[(x - 1):(x + 2), (y - 1):(y + 2)])
    #     W_two[(x - 1):(x + 2), (y - 1):(y + 2)] = 0
    #     return around
    def _get_W(self):
        W = np.zeros(shape=self.x * self.y, dtype=np.int32)
        for i in self.load_map[self.T].itertuples():
            W[i[1] * self.x + i[2]] = i[3]
        return W
    def _get_P(self):
        j = 0  ## bus
        P = np.zeros(shape=self.bus_num, dtype=np.uint8)
        for bus_bum in self.bus_map:
            temp = bus_bum.iloc[self.T]
            P[j] = temp[0] * self.x + temp[1]
            j += 1
        return P

    def get_bus_allopen(self,  sum_load):
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))

        all = sum(W_one)
        all_1_delay = 0
        all_1_energy = 0
        for index in np.arange(10):
            around_1 = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
            around_load_1 = sum(around_1.flatten())

            if (around_load_1 > self.bus_max):  # TODO
                t = around_load_1 - self.bus_max
                around_load_1 = self.bus_max
                bindex = self.obs_bus[index]
                x = int(bindex / self.x)
                y = bindex % self.x
                W_two[x, y] = t
            ## 队列上限
            # assert around_load_1 <= self.bus_max
            delay, energy = self._get_cost_bus(around_load_1)
            all_1_delay += delay * around_load_1 / sum_load
            all_1_energy += energy
        eee = all - sum(W_one)
        self.cover.append(eee)
        s1 = W_two[0:22, 0:18]
        s2 = W_two[22:, 0:18]
        s3 = W_two[0:22:, 18:]
        s4 = W_two[22::, 18:]
        s1_0_delay, s1_0_energy = self._get_cost_ecd(sum(sum(s1)))
        s2_0_delay, s2_0_energy = self._get_cost_ecd(sum(sum(s2)))
        s3_0_delay, s3_0_energy = self._get_cost_ecd(sum(sum(s3)))
        s4_0_delay, s4_0_energy = self._get_cost_ecd(sum(sum(s4)))
        new_all_0_delay = (s1_0_delay * sum(sum(s1)) + s2_0_delay * sum(sum(s2)) + s3_0_delay * sum(
            sum(s3)) + s4_0_delay * sum(sum(s4))) / sum(W_one)
        new_all_0_energy = (s1_0_energy + s4_0_energy + s3_0_energy + s2_0_energy)

        ## 无用 原来只有一个边缘服务器
        # W_load = sum(W_one.flatten())
        # delay_1, energy_1 = self._get_cost_ecd(W_load)
        # delay_1 *= W_load / sum_load

        all_1_delay += new_all_0_delay
        all_1_energy += new_all_0_energy
        return all_1_delay, all_1_energy

    def step_second(self, action):
        #TODO 遍历栅格
        self.rw_local = 0
        rw_all_0 = 0
        rw_all_1 = 0
        opentime = 1
        for t in range(opentime):
            ##read the data
            self.obs_taxi = self._get_W()

            self.obs_bus = self._get_P()

            self.obs_action = action

            ##处理负载计算
            W_one = copy.deepcopy(self.obs_taxi)
            W_two = W_one.reshape((self.y, -1))
            ##W_nonzero_index = W_one.nonzero()
            self.my_plot.load_num.append(sum(W_one))
            sum_load = sum(W_one)
            all_0_delay, all_0_energy = self._get_cost_ecd(sum_load)

            all_1_delay, all_1_energy = self.get_bus_allopen(sum_load)

            Bus_delay = 0
            Bus_energy = 0

            self.cost_place = 0

            for index, act in enumerate(action):  # TODO
                around_load = 0
                if act:
                    around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
                    around_load = sum(around.flatten())
                    ###
                    # t = 0
                    # if (around_load > 200):  #TODO
                    #     t = around_load - 200
                    #     around_load = 200
                    #     W_one[self.obs_bus[index]] = t
                    #     self.rw_local += 1
                    ###

                    delay, energy = self._get_cost_bus(around_load)
                    Bus_delay += delay * around_load / sum_load
                    Bus_energy += energy
                self.my_plot.around_list.append(around_load)
            W_load = sum(W_one.flatten())
            dqn_delay, dqn_energy = self._get_cost_ecd(W_load)
            dqn_delay *= W_load / sum_load

            cost_delay = dqn_delay + Bus_delay
            cost_energy = dqn_energy + Bus_energy
            reward = self._get_reward(cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy,
                                      all_1_energy)  # TODO

            self.my_plot.cost_delay.append([cost_delay, all_1_delay, all_0_delay])
            self.my_plot.cost_energy.append([cost_energy, all_1_energy, all_0_energy])

            self.rw_local += reward

            rw_all_0 += -self.delay_Weight * (all_0_delay - all_1_delay) / (all_0_delay - all_1_delay)
            rw_all_1 += -self.energy_Weight * (all_1_energy - all_0_energy) / (all_1_energy - all_0_energy)

            assert (all_1_energy >= all_0_energy)
            ##assert (all_1_energy >= cost_energy)
            assert (cost_delay <= all_0_delay)
            assert (all_1_delay <= all_0_delay)

            self.observation_space = self._get_obs()
            done = self._get_done()  # TODO
            info = None  # TODO
            if done:
                return copy.deepcopy(self.observation_space), self.rw_local / opentime, done, info
            ##print(self.cost_place)
            self.T += 1

        self.my_plot.rw_all_0.append(rw_all_0 / opentime)
        self.my_plot.rw_all_1.append(rw_all_1 / opentime)
        self.my_plot.reward_list.append(self.rw_local / opentime)
        return copy.deepcopy(self.observation_space), self.rw_local / opentime, done, info
        ##return copy.deepcopy(self.observation_space), reward, done, info

