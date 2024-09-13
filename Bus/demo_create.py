import geopandas as gpd
#导入TransBigData包
import transbigdata as tbd
#读取数据
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_data(bounds,params):
    path = "Bus/bus_vehicleNum/"
    # 早期
    # filename = ["01S0D034.txt", "01S0D039.txt", "01S0D040.txt", "01S0D041.txt", "01S0D042.txt", "01S0D045.txt",
    #             "01S0D047.txt", "01S0D048.txt", "01S0D105.txt", "01S0D106.txt"]
    # 毕业论文-0425
    # filename = ["04S2C080.txt", "01S2C098.txt", "02S2A034.txt", "06S0D332.txt", "02S0D093.txt", "03S2A064.txt",
    #                         "06S0D304.txt", "04S2A158.txt", "02KGP396.txt", "02S2A031.txt"]
    filename = ["01S0D034.txt", "01S2C098.txt", "02S2A034.txt", "06S0D332.txt", "02S0D093.txt", "03S2A064.txt",
                "06S0D304.txt", "04S2A158.txt", "02KGP396.txt", "02S2A031.txt"]
    # bus_load = []
    i = 0
    data_list = []
    for file in filename:
        data = pd.read_csv(path + file)
        data_list.append(data)
    data_all = pd.concat(data_list)
    tbd.data_summary(data_all,
                     col=['vehicleNum', 'time'],
                     show_sample_duration=True)
    print("end")

def create_busdata(bounds,params):
    path = "Bus/bus_vehicleNum/"
    # 早期
    # filename = ["01S0D034.txt", "01S0D039.txt", "01S0D040.txt", "01S0D041.txt", "01S0D042.txt", "01S0D045.txt",
    #             "01S0D047.txt", "01S0D048.txt", "01S0D105.txt", "01S0D106.txt"]
    #毕业论文-0425
    # filename = ["04S2C080.txt", "01S2C098.txt", "02S2A034.txt", "06S0D332.txt", "02S0D093.txt", "03S2A064.txt",
    #                         "06S0D304.txt", "04S2A158.txt", "02KGP396.txt", "02S2A031.txt"]
    filename = ["01S0D034.txt", "01S2C098.txt", "02S2A034.txt", "06S0D332.txt", "02S0D093.txt", "03S2A064.txt",
                            "06S0D304.txt", "04S2A158.txt", "02KGP396.txt", "02S2A031.txt"]
    # bus_load = []
    i = 0
    for file in filename:
        data = pd.read_csv(path + file)
        data = tbd.clean_outofbounds(data, bounds=bounds, col=['lng', 'lat'])
        data['LONCOL'], data['LATCOL'] = tbd.GPS_to_grid(data['lng'], data['lat'], params)
        tbd.data_summary(data,
                         col=['vehicleNum', 'time'],
                         show_sample_duration=True)
        order_data = data.copy()
        time = pd.to_datetime(order_data['time'])
        order_data = order_data.set_index(time)
        order_data = pd.DataFrame(order_data.resample('1T'))
        order_data = order_data[order_data[0].dt.hour.isin(np.arange(6, 18))]
        start_time = pd.to_datetime('2007-02-19 06:00:00')
        if str(order_data.iloc[0][0]) != start_time:
            add_data = data.iloc[0].copy()
            add_data['time'] = start_time
            data.iloc[0] = add_data
        time = pd.to_datetime(data['time'])
        data = data.set_index(time)
        dataset = pd.DataFrame(data.resample('1T'))
        dataset = dataset[dataset[0].dt.hour.isin(np.arange(6, 18))]
        bus_map = []
        for row in dataset.itertuples():
            a = row[2].groupby(['LONCOL', 'LATCOL'])['vehicleNum'].count().reset_index()
            if not a.empty:
                pre = row
                a = a.iloc[0]
            else:
                a = pre[2].groupby(['LONCOL', 'LATCOL'])['vehicleNum'].count().reset_index()
                a = a.iloc[0]
            a['time'] = row[1]
            bus_map.append(a)
        bus = pd.DataFrame(bus_map)
        bus['geometry'] = tbd.grid_to_polygon([bus['LONCOL'], bus['LATCOL']], params)
        bus = gpd.GeoDataFrame(bus)
        # 绘制
        bus.plot(column='vehicleNum')
        plt.show()
        bus.to_csv("Bus/real_time_bus/bus_" + str(i) + ".csv", index=False)
        i += 1
    print("end")