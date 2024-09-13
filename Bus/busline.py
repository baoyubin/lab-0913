import warnings
warnings.filterwarnings('ignore')
#导入TransBigData包
import transbigdata as tbd
#读取数据
import pandas as pd


busdata, stop = tbd.getbusdata('shanghai', '66', accurate=True)
data = busdata['geometry'][0]
busdata.plot()