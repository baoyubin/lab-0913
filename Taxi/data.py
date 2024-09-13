
import pandas as pd


import os
def taxi_data():
    path = 'taxi_number/'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            data = pd.read_csv('taxi_number/%s' % file, header=None)
            data.columns = ['VehicleNum', 'time', 'slon', 'slat', 'Speed', 'Speed', 'OpenStatus', ]
            print('%s : ' % file + str(max(data['slat'])) + '，' + str(min(data['slat'])))
            assert max(data['slon']) <= 122.0, min(data['slon']) >= 121.0
            assert max(data['slat']) <= 32.0, min(data['slat']) >= 30.9

