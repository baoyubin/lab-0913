import pandas as pd
import transbigdata as tbd
#创建图框
import matplotlib.pyplot as plt
import read_data as rd

def printmap():
    # 如果是windows系统，路径这么写，最后注意要两个斜杠以防转义
    tbd.set_imgsavepath(r'D:\PycharmProject\my_dqn\\')
    # 定义显示范围范围
    bounds = [121.0, 30.9, 122.0, 32.0]
    fig = plt.figure(1, (8, 8), dpi=250)
    ax = plt.subplot(111)
    plt.sca(ax)
    # 添加地图底图
    tbd.plot_map(plt, bounds, zoom=12, style=0)
    # 添加比例尺和指北针
    tbd.plotscale(ax, bounds=bounds, textsize=10, compasssize=1, accuracy=2000, rect=[0.06, 0.03], zorder=10)
    plt.axis('off')
    plt.xlim(bounds[0], bounds[2])
    plt.ylim(bounds[1], bounds[3])
    plt.show()

class Load_map:
    def __init__(self):
        self.load_map = []
    def get_map(self, bounds, params):
        rd.read_taxi(self.load_map, bounds, params)


class Bus_map:
    def __init__(self):
        self.bus_map = []
        self.path = "Bus/real_time_bus/bus_"
    def get_map(self, bounds, params):
        i = 0
        while i < 10:
            self.bus_map.append(pd.read_csv(self.path + str(i) + ".csv"))
            i += 1



