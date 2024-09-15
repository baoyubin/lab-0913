import transbigdata as tbd
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np
from transbigdata import sample_duration


def show_taxidata(bounds,params):
    path = 'Taxi/taxi_demo/'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称\
    data_list = []
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            data = pd.read_csv(path + file, header=None)
            data.columns = ['VehicleNum', 'time', 'lng', 'lat', 'Speed', 'Speed', 'OpenStatus']
            data_list.append(data)
    data_all = pd.concat(data_list)
    data_summary(data_all,
                     col=['VehicleNum', 'time'],
                     show_sample_duration=True,
                     roundnum=3)

def create_taxidata(bounds,params):
    path = 'Taxi/taxi_demo/'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称\
    data_list = []
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            data = pd.read_csv(path + file, header=None)
            data.columns = ['VehicleNum', 'time', 'lng', 'lat', 'Speed', 'Speed', 'OpenStatus']
            data = tbd.clean_taxi_status(data, col=['VehicleNum', 'time', 'OpenStatus'])
            ## 远程主机为Time
            ##data = tbd.clean_taxi_status(data, col=['VehicleNum', 'Time', 'OpenStatus'])
            data = tbd.clean_outofbounds(data, bounds, col=['lng', 'lat'])
            ## data = [data['VehicleNum'], data['time'], data['lng'], data['lat']]
            data['LONCOL'], data['LATCOL'] = tbd.GPS_to_grid(data['lng'], data['lat'], params)
            time = pd.to_datetime(data['time'])
            data = data.set_index(time)
            # 计算LONCOL等于5且LATCOL等于45的行数
            count = sum((data['LONCOL'] == 5) & (data['LATCOL'] == 44))
            cnt = sum((data['LONCOL'] == 20) & (data['LATCOL'] == 34))
            c = sum((data['LONCOL'] == 28) & (data['LATCOL'] == 36))
            cd = sum((data['LONCOL'] == 27) & (data['LATCOL'] == 42))
            cde = sum((data['LONCOL'] == 25) & (data['LATCOL'] == 16))
            if (count <= 400 and cnt <= 400 and c <=200 and cd <= 200 and cde <= 30):
                data_list.append(data)

    data_all = pd.concat(data_list)
    data_all['geometry'] = tbd.grid_to_polygon([data_all['LONCOL'], data_all['LATCOL']], params)
    data_all = gpd.GeoDataFrame(data_all)
    data_all.to_csv("Taxi/real_time load/taxi.csv", index=False)

def data_summary(data, col=['Vehicleid', 'Time'], show_sample_duration=False,
                 roundnum=4):
    '''
    Output the general information of the dataset.

    Parameters
    -------
    data : DataFrame
        The trajectory points data
    col : List
        The column name, in the order of [‘Vehicleid’, ‘Time’]
    show_sample_duration : bool
        Whether to output individual sampling interval
    roundnum : number
        Number of decimal places
    '''
    [Vehicleid, Time] = col
    print('Amount of data')
    print('-----------------')
    print('Total number of data items: ', len(data))
    Vehicleid_count = data[Vehicleid].value_counts()
    print('Total number of individuals: ', len(Vehicleid_count))
    print('Data volume of individuals(Mean): ',
          round(Vehicleid_count.mean(), roundnum))
    print('Data volume of individuals(Upper quartile): ',
          round(Vehicleid_count.quantile(0.75), roundnum))
    print('Data volume of individuals(Median): ', round(
        Vehicleid_count.quantile(0.5), roundnum))
    print('Data volume of individuals(Lower quartile): ',
          round(Vehicleid_count.quantile(0.25), roundnum))
    print('')
    print('Data time period')
    print('-----------------')
    print('Start time: ', data[Time].min())
    print('End time: ', data[Time].max())
    print('')
    if show_sample_duration:
        sd = sample_duration(data, col=[Vehicleid, Time])
        print('Sampling interval')
        print('-----------------')
        print('Mean: ', round(sd['duration'].mean(), roundnum), 's')
        print('Upper quartile: ', round(
            sd['duration'].quantile(0.75), roundnum), 's')
        print('Median: ', round(sd['duration'].quantile(0.5), roundnum), 's')
        print('Lower quartile: ', round(
            sd['duration'].quantile(0.25), roundnum), 's')
        # Plot the distribution of sampling interval
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        fig = plt.figure(1,(8,3),dpi=300)
        ax = plt.subplot(111)
        plt.subplots_adjust(left=0.1,right=0.98,top=0.9,bottom=0.19)
        sns.kdeplot(sd[sd['duration']<sd['duration'].quantile(0.95)]['duration'],ax=ax,legend=False)
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,0))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.xlabel('采样间隔(s)')
        plt.ylabel('密度')
        plt.xlim(0, sd['duration'].quantile(0.95))
        plt.savefig('出租车采集数据' + '.svg', format='svg')
        plt.show()
