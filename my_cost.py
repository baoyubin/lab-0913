import math


class Cost:
    def __init__(self):


        ## W
        self.P_t = 0.25
        ##结果
        self.N = 10 ** (-14)

        ##self.g = 127 + 30 * math.log(1, 2)D
        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        ## bit
        self.Data_bit = 10 * self.KB

        ## CPU cycles per bit
        self.computational_intensity = 50
        self.computational_data_cycle = self.computational_intensity * self.Data_bit

    ## bit/s
    def get_transmission_rate(self, B, LOAD_NUM, g):
        log = math.log(1 + self.P_t * g / self.N, 2)
        return B / LOAD_NUM * log

    ## in CPU cycles per bit（computational_intensity）
    def get_computation_delay(self, LOAD_NUM, CPU_frequency):
        avr_CPU_frequency = CPU_frequency / LOAD_NUM
        time = self.computational_data_cycle / avr_CPU_frequency
        return time

    def get_transmission_delay(self, B, LOAD_NUM, g):
        transmission_rate = self.get_transmission_rate(B, LOAD_NUM, g)
        return self.Data_bit / transmission_rate
