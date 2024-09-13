import numpy as np
import copy
import math
from my_cost import Cost
class Env:
    def __init__(self, x, y, load_map, bus_map, my_plot, delay_Weight=0.2, idle_w=0.5, data_num=3):
        self.x = x
        self.y = y
        self.T = 0
        self.cost_func = Cost()

        ##plot
        self.my_plot = my_plot
        self.rw_local = 0

        self.bus_num = 10
        self.action_num = 10
        self.bus_bound = 5

        self.bus_map = bus_map
        self.load_map = load_map

        self.delay_Weight = delay_Weight
        self.energy_Weight = 1 - self.delay_Weight

        self.idle_w = idle_w


        ## HZ 10 * 10**6
        self.B_bus = 10 * 10**6
        self.B_ecd = 4 * self.B_bus
        ## HZ 2*10**9
        self.BUS_CPU_frequency = 2 * 10**9
        self.ECD_CPU_frequency = 4 * self.BUS_CPU_frequency
        self.cost_place = 0

        ##queue TODO
        self.queue = []
        self.T_FINISH = 1000
        self.data_num = data_num
        self.service_rate = 600

        self.P_bus = 10 ##W
        self.P_ecd = 20
        self.observation_space = dict(
            {
               "taxi": np.zeros(shape=(self.x*self.y), dtype=np.int32), ##environment
               "bus": np.zeros(shape=self.bus_num, dtype=np.int32), ##environment

            }
        )

        ##self.action_space = np.random.randint(0, 2, self.bus_num) ##action[0 1]
        self.action_space = np.zeros(shape=self.bus_num, dtype=np.int32)
        self.obs_taxi = self.observation_space["taxi"]
        self.obs_bus = self.observation_space["bus"]

    def reset(self):
        self.T = 0
        self.obs_taxi = self._get_W()
        self.obs_bus = self._get_P()
        ##self.T = 1
        self.obs_action = self.action_space
        self.observation_space = self._get_obs()
        return copy.deepcopy(self.observation_space)

    def _get_obs(self):
        return {"taxi": self.obs_taxi, "bus": self.obs_bus}

    def _get_cost_bus(self, off_sum): #TODO
        idle = 0
        if(off_sum == 0):
            idle = self.P_bus * self.idle_w * 60
            return 0, 0 + idle
        off_sum *= self.data_num
        computingDelay_Bus = self.cost_func.get_computation_delay(off_sum, self.BUS_CPU_frequency)
        assert (computingDelay_Bus <= 60)
        exe_energy = computingDelay_Bus * self.P_bus
        cost_trantoBus = self.cost_func.get_transmission_delay(self.B_bus, off_sum)
        tran_energy = cost_trantoBus * self.cost_func.P_t * off_sum

        idle = self.P_bus * self.idle_w * (60 - computingDelay_Bus)

        return cost_trantoBus + computingDelay_Bus, tran_energy + exe_energy + idle

    def _get_cost_ecd(self, load_sum):
        if(load_sum == 0):
            return 0, 0

        cost_queue = 0
        if (load_sum > self.service_rate):
            λ = load_sum - self.service_rate
            cost_queue = λ / (2 * self.service_rate * (self.service_rate - λ))
        load_sum *= self.data_num
        ## 计算延迟 =  每个任务计算延迟
        computingDelay_ECD = self.cost_func.get_computation_delay(load_sum, self.ECD_CPU_frequency)
        exe_energy = computingDelay_ECD * self.P_ecd
        ## 传输时间 = 卸载任务数量 * 每个任务传输延迟
        cost_trantoEcd = self.cost_func.get_transmission_delay(self.B_ecd, load_sum)
        tran_energy = cost_trantoEcd * self.cost_func.P_t * load_sum

        assert (cost_queue == 0)
        return computingDelay_ECD + cost_trantoEcd + cost_queue, tran_energy

    def _get_reward(self, cost_delay, all_0_delay, all_1_delay, cost_energy, all_0_energy, all_1_energy): #TODO
        self.my_plot.avg_delay.append(cost_delay)
        ##return (cost_offalltoecd - obs_offloadtoBus - obs_offloadtoEcd) / cost_offalltoecd - self.rw_local / 10 * 0.3
        ##return (cost_offalltoecd - obs_offloadtoBus - obs_offloadtoEcd)
        #return -0.5*(obs_offloadtoBus+obs_offloadtoEcd)-0.1*self.cost_place


        delay = (cost_delay - all_1_delay) / (all_0_delay - all_1_delay) ##all_1 不一定是最小延迟
        energy = (cost_energy - all_0_energy) / (all_1_energy - all_0_energy)

        ##delay = math.log(cost_delay, 10) / math.log(all_0_delay, 10)
        ##energy = math.log(cost_energy, 10) / math.log(all_1_energy, 10)

        rew = - self.delay_Weight * delay - self.energy_Weight * energy
        return rew

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
        opentime = 10
        for t in range(opentime):
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

            #3- self.delay_Weight * delay - self.energy_Weight * energy
            self.rw_local += reward

            rw_all_0 += -self.delay_Weight * (all_0_delay - all_1_delay) / (all_0_delay - all_1_delay)
            ##rw_all_0 += -self.delay_Weight * math.atan2(all_0_delay, 1) / math.pi * 2 - \
            ##           self.energy_Weight * math.atan2(all_0_energy, 1) / math.pi * 2
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
        all_1_delay = 0
        all_1_energy = 0
        for index in np.arange(10):
            around_1 = self.get_around(self.obs_bus[index], W_two, self.bus_bound, True)
            around_load_1 = sum(around_1.flatten())
            delay, energy = self._get_cost_bus(around_load_1)
            all_1_delay += delay * around_load_1 / sum_load
            all_1_energy += energy
        W_load = sum(W_one.flatten())
        delay_1, energy_1 = self._get_cost_ecd(W_load)
        delay_1 *= W_load / sum_load

        all_1_delay += delay_1
        all_1_energy += energy_1

        return all_1_delay, all_1_energy