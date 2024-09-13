import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_default(map, ylab='Reward' ):
    name = ['全部开启', '全部关闭', 'FTS', 'DQN-OD']

    plt.plot(map[1],marker='d',label =name[0])
    plt.plot(map[2],marker='>', label=name[1])
    plt.plot(map[3], marker='<', label=name[2], linestyle = '--')
    plt.plot(map[0], marker='o', label=name[3])
    # plt.xlabel('场景')
    plt.xlabel('用户数量')
    plt.ylabel(ylab)
    # plt.xticks([0, 1, 2, 3], ['场景1', '场景2', '场景3', '场景4'])
    plt.xticks([0, 1, 2, 3], ['600', '750', '900', '1050'])
    plt.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.savefig(ylab+'.svg', format='svg')
    plt.show()

def plot_dou(map, a=None, ylab='Reward' ):



    # 创建图表和第一个y轴
    fig, ax1 = plt.subplots()

    # 绘制第一个y轴的数据
    color = 'tab:red'
    ax1.set_ylabel(ylab)
    ax1.set_xlabel('Time Frames')
    ax1.tick_params(axis='y', labelcolor=color)
    s = "ygb"
    name = ['总能耗', '归一化时延']
    # 创建第二个y轴，共享相同的x轴
      # 创建次y轴

    ax1.plot(np.arange(len(map)), map)
    if (a != None):
        ax2 = ax1.twinx()
        ax2.set_ylabel('action', color=color)  # 设置次y轴的标签
        ax2.plot(np.arange(len(a)), a, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.show()
def min_max_scaler(data,max_value,min_value):
    return [(x - min_value) / (max_value - min_value) for x in data]
if __name__ == '__main__':
    ##实验结果1
    # delay = [[0.31888362756092625, 0.4429155548560186, 0.5925688884605638, 0.6875283479813833], [0.29859860673863653, 0.39925803515418984, 0.5393971319578796, 0.6591483017328995], [0.4738705969518146, 0.7791451826683755, 0.9710442955312352, 1.088370960721589]]
    delay = [[0.32071178747574813, 0.4494874216587472, 0.5878361002484515,  0.7195897696828608], [0.29760839111636195, 0.3979803308040832, 0.53785710831519, 0.6574188242189402], [0.4724338287004145, 0.7772783055236304, 0.9687879328677281, 1.0857815493930787], [0.3003313553309171, 0.4008416522266735, 0.5405053153673842, 0.6579836949853898]]
    delay[0] = [item * 1000 for item in delay[0]]
    delay[1] = [item * 1000 for item in delay[1]]
    delay[2] = [item * 1000 for item in delay[2]]
    delay[3] = [item * 1000 for item in delay[3]]
    # energy = [[406.53545069167757, 421.48268699844135, 453.5845159194505, 476.2291628323112], [544.426999884933, 559.8369592268941, 575.7357400141752, 588.6102321124964], [253.11575005066697, 274.4112919178421, 297.05476723264115, 317.2240971129386], [719.1305695927856, 813.605513037248, 856.3622961698086, 950.6840351923307]]
    energy = [[218.08550610051623, 236.43811355977, 258.83492118390575, 279.53304436282934], [266.50787066934913, 283.2453446600098, 300.6970535196884, 314.87671382266564], [177.28082826686025, 201.315809402236, 227.13027535486185, 250.35493157956782], [259.8911736205285, 280.0581936318564, 298.60848301789133, 314.5072225715108]]
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


    # delay = [1.0563202937541067, 1.0463202937541067, 0.8682059491130265, 0.794719613238435, 0.7643502844540371,0.7547272143355528, 0.7371048187391924, 0.7244182110154899, 0.7275283479813833]
    # energy = [325.123213414131, 331.31245782908104, 383.09266656833853, 418.5842682148504, 422.5414240835018,434.03311641686014, 447.5484114221771, 449.9958577670207, 454.2291628323112]
    # plot_dou(delay, energy)