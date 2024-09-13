import geopandas as gpd
#导入TransBigData包
import transbigdata as tbd
#读取数据
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('taxi_number/Taxi_105', header=None)
data.columns = ['VehicleNum', 'time', 'lng', 'lat', 'Speed', 'Speed', 'OpenStatus']

# 数据预处理
# 剔除出租车数据中载客状态瞬间变化的记录
data = tbd.clean_taxi_status(data, col=['VehicleNum', 'time', 'OpenStatus'])

# 栅格化
# 定义范围，获取栅格化参数
#bounds = [121.0, 30.9, 122.0, 32.0]
bounds = [121.4, 31.15, 121.6, 31.35]
params = tbd.area_to_params(bounds, accuracy=500, method='rect')

# 将GPS栅格化
data['LONCOL'], data['LATCOL'] = tbd.GPS_to_grid(data['lng'], data['lat'], params)

# 集计栅格数据量
datatest = data.groupby(['LONCOL', 'LATCOL'])['VehicleNum'].count().reset_index()

# 生成栅格地理图形
grid = [datatest['LONCOL'], datatest['LATCOL']]
datatest['geometry'] = tbd.grid_to_polygon(grid, params)
# 转为GeoDataFrame
datatest = gpd.GeoDataFrame(datatest)

# 绘制
datatest.plot(column='VehicleNum')
plt.show()
