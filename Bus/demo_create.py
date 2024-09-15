import geopandas as gpd
#导入TransBigData包
import transbigdata as tbd
#读取数据
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transbigdata import sample_duration


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
    data_summary(data_all,
                     col=['vehicleNum', 'time'],
                     show_sample_duration=True,
                     roundnum=3)
    print("end")

def create_busdata(bounds,params):
    path = "Bus/bus_vehicleNum/"
    # 早期
    # filename = ["01S0D034.txt", "01S0D039.txt", "01S0D040.txt", "01S0D041.txt", "01S0D042.txt", "01S0D045.txt",
    #             "01S0D047.txt", "01S0D048.txt", "01S0D105.txt", "01S0D106.txt"]
    #毕业论文-0425
    # filename = ["04S2C080.txt", "01S2C098.txt", "02S2A034.txt", "06S0D332.txt", "02S0D093.txt", "03S2A064.txt",
    #                         "06S0D304.txt", "04S2A158.txt", "02KGP396.txt", "02S2A031.txt"]
    filename = ["01S0D039.txt", "01S2C098.txt", "02S2A034.txt", "06S0D332.txt", "02S0D093.txt", "03S2A064.txt",
                            "06S0D304.txt", "04S2A158.txt", "02KGP396.txt", "02S2A031.txt"]
    # bus_load = []
    i = 0
    for file in filename:
        data = pd.read_csv(path + file)
        data = tbd.clean_outofbounds(data, bounds=bounds, col=['lng', 'lat'])
        data['LONCOL'], data['LATCOL'] = tbd.GPS_to_grid(data['lng'], data['lat'], params)

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
        bus_all = []
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
        bus_all.append(bus)
        i += 1

    print("end")
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
        plt.savefig('公交车采集数据' + '.svg', format='svg')
        plt.show()
