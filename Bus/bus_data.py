import pandas as pd

import os
def bus_data():
    path = 'bus_vehicleNum/'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    maxlng = 0
    maxlat = 0
    minlng = 99999
    minlat = 99999
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            data = pd.read_csv(path + file)
            count = data.iloc[:, 0].size
            if(count > 500):
                maxlat = max(data['lat']) if max(data['lat']) > maxlat else maxlat
                minlat = min(data['lat']) if min(data['lat']) < minlat else minlat
                maxlng = max(data['lng']) if max(data['lng']) > maxlng else maxlng
                minlng = min(data['lng']) if min(data['lng']) < minlng else minlng
    print(maxlng, minlng, maxlat, minlat)
    ##121.911745 121.00013 31.589883 30.721725

if __name__ == '__main__':
    bus_data()