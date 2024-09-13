import transbigdata as tbd
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np

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
    tbd.data_summary(data_all,
                     col=['VehicleNum', 'time'],
                     show_sample_duration=True)

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
            data_list.append(data)
    data_all = pd.concat(data_list)
    data_all['geometry'] = tbd.grid_to_polygon([data_all['LONCOL'], data_all['LATCOL']], params)
    data_all = gpd.GeoDataFrame(data_all)
    data_all.to_csv("Taxi/real_time load/taxi.csv", index=False)