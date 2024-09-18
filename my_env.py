import numpy as np
import copy
import math

from MapConfig import MapConfig
from my_cost import Cost
class Env:
    def __init__(self, x, y, load_map, bus_map, my_plot, delay_Weight=0.5, idle_w=0.5, data_num=1):
        self.all_load = np.zeros(shape=(x * y), dtype=np.int32).reshape((y, -1))
        self.mapConfig = MapConfig()
        self.bus_load = [0] * 10

        self.s = 4
        # 基本参数
        # 频率
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz

        self.nor = 10 ** (-7)
        self.nor1 = 10 ** 19

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB


        self.cover =[]

        self.eswitch = 0
        self.ecdidle = 83
        self.busidle = 83
        self.opentime = 10
        self.bus_bound = 6
        self.x = x
        self.y = y
        self.T = 0
        self.cost_func = Cost()

        ##结果
        self.g_bus = 4.11 * (3 * 10 ** 2 / (4 * math.pi * 915 * self.bus_bound*self.mapConfig.gripAc)) ** (2.8)
        self.g_mec = 4.11 * (3 * 10 ** 2 / (4 * math.pi * 915 * 2 * self.bus_bound*self.mapConfig.gripAc)) ** (2.8)
        self.maxload = 0



        self.ae = 0.001
        self.at = 5
        self.maxdelay = 1.5 #1.5
        self.mindelay = 0
        self.maxenergy = 400 #400
        self.minenergy = 140 #论文结果2

        ##plot
        self.my_plot = my_plot
        self.rw_local = 0

        # busbound=5
        self.bus_num = 10
        self.action_num = 10


        self.bus_map = bus_map
        self.load_map = load_map

        self.delay_Weight = delay_Weight
        self.energy_Weight = 1 - self.delay_Weight

        self.idle_w = idle_w


        self.B_bus = 20 * self.mHz
        self.B_ecd = 20 * self.mHz
        self.B_cloud = self.B_ecd

        self.BUS_CPU_frequency = 10 * self.GHz
        self.ECD_CPU_frequency = 20 * self.GHz
        self.Cloud_CPU_frequency = 20 * self.GHz

        self.cost_place = 0

        ##queue TODO
        self.queue = []
        self.T_FINISH = 500
        self.data_num = data_num
        self.service_rate = 20
        self.bus_max = 20 ## 真实队列为*datanum
        self.P_bus = 156 ##W
        self.P_ecd = 312
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
        assert off_sum <= self.bus_max
        if(off_sum == 0):
            return 0, self.busidle + (self.P_bus - self.busidle) * off_sum / self.bus_max
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

        energy = self.busidle + (self.P_bus - self.busidle) * off_sum / self.bus_max
        return cost_trantoBus + computingDelay_Bus, tran_energy + energy

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


    def new_get_ecd(self, load_sum):
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
        cloud_size = 0
        if (load_sum > self.service_rate):
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
        computingDelay_ECD, exe_energy = self.ecd_not_cloud(allsum)
        delay = computingDelay_ECD + cost_trantoEcd
        energy = self.ecdidle + (self.P_ecd - self.ecdidle) * allsum / self.service_rate
        return delay + cost_queue, tran_energy + energy

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
        return self.new_get_ecd(load_sum)
        return self.cloud_get_ecd(load_sum=load_sum)
        # self.maxload = max(self.maxload,load_sum) ## 246
        # self.cloud_get_ecd(load_sum)
        allsum = load_sum * self.data_num
        load_sum *= self.data_num
        if(load_sum == 0):
            return 0, 0
        cost_queue = 0
        ## cost_trantoEcd为平均传输时间
        cost_trantoEcd = self.cost_func.get_transmission_delay(self.B_ecd, allsum, self.g_mec)
        ## 总传输时间 = 卸载任务数量 * 每个任务传输延迟 cost_trantoEcd * load_sum
        ## 总传输能耗 tran_energy = cost_trantoEcd * self.cost_func.P_t * load_sum
        tran_energy = cost_trantoEcd * self.cost_func.P_t * allsum
        if (load_sum > self.service_rate):
            # 2 * self.service_rate - load_sum > 0
            if 2 * self.service_rate - load_sum > 0:
                cost_queue = (load_sum - self.service_rate) / (2 * self.service_rate * (2 * self.service_rate - load_sum))
            else:
                ecd_size = 2 * self.service_rate - 1
                cost_queue = (ecd_size - self.service_rate) / (
                        2 * self.service_rate * (2 * self.service_rate - ecd_size))
                cloud_size = allsum - ecd_size
                print("error:cost,queue")
        computingDelay_ECD, exe_energy = self.ecd_not_cloud(allsum)
        return computingDelay_ECD + cost_trantoEcd + cost_queue, tran_energy

    def _get_reward_s(self, cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy, all_1_energy,
                    action):  # TODO
        self.my_plot.avg_delay.append(cost_delay)

        delay = (cost_delay - all_1_delay) / (all_0_delay - all_1_delay)
        cost = cost_energy / 3000
        assert cost < 1
        rew = (1 - self.delay_Weight) * cost /self.delay_Weight * delay
        ra = (1 - self.delay_Weight) * cost - self.delay_Weight * delay
        return ra


    def _get_reward(self, cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy, all_1_energy, action, topk_delay, topk_energy): #TODO
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



        delay = (cost_delay - self.mindelay) / (self.maxdelay - self.mindelay)
        energy = (cost_energy - self.minenergy) / (self.maxenergy - self.minenergy)
        rew = (- self.delay_Weight * delay - self.energy_Weight * energy)
        # rew = delay/energy
        rew = - self.delay_Weight * cost_delay * self.at - self.energy_Weight * cost_energy * self.ae
        r0 = - self.delay_Weight * all_0_delay * self.at - self.energy_Weight * all_0_energy * self.ae
        r1 = - self.delay_Weight * all_1_delay * self.at - self.energy_Weight * all_1_energy * self.ae
        r2 = - self.delay_Weight * topk_delay * self.at - self.energy_Weight * topk_energy * self.ae
        ## todo 改成 min=0 max=x，后续根据值进行更新最大值
        # if rew > 0:
        #     return 0
        ## assert rew > 0
        self.my_plot.maxmindelay.append(delay)
        self.my_plot.maxminenergy.append(energy)
        return rew, r0, r1, r2

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
            self.all_load += W_two

            self.my_plot.load_num.append(sum(W_one))
            sum_load = sum(W_one)

            all_0_delay, all_0_energy = self.compute_area(W_two, sum_load)
            all_1_delay, all_1_energy, sorted_indices = self.get_bus_allopen(sum_load)



            topk = sum(action)
            new_list = [0] * 10
            for i in range(topk):
                new_list[sorted_indices[i]] = 1
            topk_delay, topk_energy = self.get_bus_topk(sum_load, np.array(new_list))

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
            dqn_delay, dqn_energy = self.compute_area(W_two, sum_load)

            cost_delay = dqn_delay + Bus_delay
            cost_energy = dqn_energy + Bus_energy
            if (t==0):
                cost_energy += num * self.eswitch


            reward, rw_all_0, rw_all_1, rw_topk = self._get_reward(cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy,
                                      all_1_energy, action, topk_delay, topk_energy)  # TODO

            if (reward > rw_topk):
                reward = 20 * (reward - rw_topk)
            else:
                reward = 20 * (reward - rw_topk)
            self.my_plot.cost_delay.append([cost_delay, all_1_delay, all_0_delay])
            self.my_plot.cost_energy.append([cost_energy, all_1_energy, all_0_energy])

            #3- self.delay_Weight * delay - self.energy_Weight * energy
            self.rw_local += reward

            # rw_all_0 += -self.delay_Weight * (all_0_delay - all_1_delay) / (all_0_delay - all_1_delay)
            # rw_all_1 += -self.energy_Weight * (all_1_energy - all_0_energy) / (all_1_energy - all_0_energy)
            # rw_all_0 += - self.delay_Weight * all_0_delay * self.at - self.energy_Weight * all_0_energy * self.ae
            # rw_all_1 += - self.delay_Weight * all_1_delay * self.at - self.energy_Weight * all_1_energy * self.ae
            # rw_all_0 += - self.delay_Weight * (all_0_delay - self.mindelay) / (self.maxdelay - self.mindelay) - self.energy_Weight * (all_0_energy - self.minenergy) / (self.maxenergy - self.minenergy)
            # rw_all_1 += - self.delay_Weight * (all_1_delay - self.mindelay) / (self.maxdelay - self.mindelay) - self.energy_Weight * (all_1_energy - self.minenergy) / (self.maxenergy - self.minenergy)

            # assert (all_1_energy >= all_0_energy)
            ##assert (all_1_energy >= cost_energy)
            ##assert (cost_delay <= all_0_delay)
            # assert (all_1_delay <= all_0_delay)
            # assert (cost_delay <= self.maxdelay and all_0_delay <= self.maxdelay)
            # assert (cost_delay >= self.mindelay and all_0_delay >= self.mindelay)
            # assert (cost_energy <= self.maxenergy and all_1_energy <= self.maxenergy)
            # assert (cost_energy >= self.minenergy and all_0_energy >= self.minenergy)

            self.observation_space = self._get_obs()
            done = self._get_done()  # TODO
            info = None  # TODO
            if done:
                return copy.deepcopy(self.observation_space), self.rw_local, done, info
            ##print(self.cost_place)
            self.T += 1

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

    def _get_W(self):
        W = np.zeros(shape=self.x * self.y, dtype=np.int32)
        for i in self.load_map[self.T].itertuples():
            # W[i[1] * self.x + i[2]] = i[3]
            W[i[2] * self.x + i[1]] = i[3]
        return W
    def _get_P(self):
        j = 0  ## bus
        P = np.zeros(shape=self.bus_num, dtype=np.uint8)
        for bus_bum in self.bus_map:
            temp = bus_bum.iloc[self.T]
            ## lat:temp[1]
            # P[j] = temp[0] * self.x + temp[1]
            P[j] = temp[1] * self.x + temp[0]
            j += 1
        return P

    def get_bus_allopen(self,  sum_load):
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))

        all_1_delay = 0
        all_1_energy = 0
        random_indices = np.random.permutation(10)
        load_action = []
        for index in random_indices:
            around_1 = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
            around_load_1 = sum(around_1.flatten())
            load_action.append(around_load_1)
            if (around_load_1 > self.bus_max):  # TODO
                t = around_load_1 - self.bus_max
                around_load_1 = self.bus_max
                bindex = self.obs_bus[index]
                x = int(bindex / self.x)
                y = bindex % self.x
                W_two[x, y] = t
            delay, energy = self._get_cost_bus(around_load_1)
            self.bus_load[index] += around_load_1
            all_1_delay += delay * around_load_1 / sum_load
            all_1_energy += energy

        eee = sum_load - sum(W_one)
        if (len(self.cover) <= 720):
            self.cover.append(eee/sum_load)
        new_all_0_delay, new_all_0_energy = self.compute_area(W_two, sum_load)
        all_1_delay += new_all_0_delay
        all_1_energy += new_all_0_energy

        sorted_indices = [idx for idx, val in sorted(enumerate(load_action), key=lambda x: x[1],reverse=True)]
        return all_1_delay, all_1_energy,sorted_indices

    def get_bus_topk(self,  sum_load, action):
        W_one = copy.deepcopy(self.obs_taxi)
        W_two = W_one.reshape((self.y, -1))
        Bus_delay = 0
        Bus_energy = 0
        for index, act in enumerate(action):  # TODO
            around_load = 0
            if act:
                around = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
                around_load = sum(around.flatten())
                ###
                if (around_load > self.bus_max):  # TODO
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
        dqn_delay, dqn_energy = self.compute_area(W_two, sum_load)

        cost_delay = dqn_delay + Bus_delay
        cost_energy = dqn_energy + Bus_energy


        return cost_delay, cost_energy

    def compute_area(self, W_two, sum_load):
        s = self.s
        ## 只支持4个
        rows, cols = W_two.shape
        num = int(s/2)
        block_size = max(rows, cols) // num
        delays = []
        energies = []
        total_delay = 0
        load = 0
        t = 0
        for i in range(num):
            for j in range(num):
                start_row = i * block_size
                if (i == num-1):
                    end_row = rows
                else:
                    end_row = start_row + block_size

                start_col = j * block_size
                if (j == num - 1):
                    end_col = cols
                else:
                    end_col = min(start_col + block_size, cols)
                sub_matrix = W_two[start_row:end_row, start_col:end_col]
                delay, energy = self._get_cost_ecd(sum(sum(sub_matrix)))
                delays.append(delay)
                total_delay = total_delay + delay * sum(sum(sub_matrix))
                energies.append(energy)
                load += sum(sum(sub_matrix))
                t += 1
        assert t == s
        assert sum(sum(W_two)) == load
        average_delay = total_delay / sum_load
        total_energy = sum(energies)
        return average_delay, total_energy


