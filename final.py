import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_default(map, ylab='Reward' ):
    name = ['全部开启', '全部关闭', 'TOP', 'DQN-OD']

    plt.plot(map[1],marker='d',label =name[0])
    plt.plot(map[2],marker='>', label=name[1])
    plt.plot(map[3], marker='<', label=name[2], linestyle = '--')
    plt.plot(map[0], marker='o', label=name[3])
    # plt.xlabel('场景')
    plt.xlabel('用户数量')
    plt.ylabel(ylab)
    plt.xticks([0, 1, 2, 3], ['600', '750', '900', '1050'])
    plt.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.savefig(ylab+'.svg', format='svg')
    plt.show()

def min_max_scaler(data,max_value,min_value):
    return [(x - min_value) / (max_value - min_value) for x in data]
if __name__ == '__main__':
    ##实验结果1
    delay = [[0.32071178747574813, 0.4494874216587472, 0.5878361002484515,  0.6711098230326941], [0.29760839111636195, 0.3979803308040832, 0.53785710831519, 0.6057569371011928], [0.4724338287004145, 0.7772783055236304, 0.9687879328677281, 1.0021940514393854], [0.3003313553309171, 0.4008416522266735, 0.5405053153673842, 0.6712166892635497]]
    delay[0] = [item * 1000 for item in delay[0]]
    delay[1] = [item * 1000 for item in delay[1]]
    delay[2] = [item * 1000 for item in delay[2]]
    delay[3] = [item * 1000 for item in delay[3]]

    energy = [[218.08550610051623, 236.43811355977, 258.83492118390575, 2479.7388194444443], [266.50787066934913, 283.2453446600098, 300.6970535196884, 2862.769930555555], [177.28082826686025, 201.315809402236, 227.13027535486185, 2177.471875], [259.8911736205285, 280.0581936318564, 298.60848301789133, 2479.929791666667]]
    energy[0] = [item * 720 / 10000 for item in energy[0]]
    energy[1] = [item * 720 / 10000 for item in energy[1]]
    energy[2] = [item * 720 / 10000 for item in energy[2]]
    energy[3] = [item * 720 / 10000 for item in energy[3]]
    percentage_improvements = []
    for v1, v2 in zip(delay[2], delay[0]):
        if v2 != 0:  # 避免除以零的错误
            improvement = (v1 - v2) / v1 * 100
            percentage_improvements.append(improvement)
    for v1, v2 in zip(delay[0], delay[1]):
        if v2 != 0:  # 避免除以零的错误
            improvement = (v1 - v2) / v1 * 100
            percentage_improvements.append(improvement)
    for v1, v2 in zip(energy[1], energy[0]):
        if v2 != 0:  # 避免除以零的错误
            improvement = (v1 - v2) / v1 * 100
            percentage_improvements.append(improvement)

    for v1, v2 in zip(delay[0], delay[3]):
        if v2 != 0:  # 避免除以零的错误
            improvement = (v1 - v2) / v1 * 100
            percentage_improvements.append(improvement)
    for v1, v2 in zip(energy[3], energy[0]):
        if v2 != 0:  # 避免除以零的错误
            improvement = (v1 - v2) / v1 * 100
            percentage_improvements.append(improvement)
    plot_default(delay, "平均时延(ms)")
    plot_default(energy, "总能耗(10^4 J)")


