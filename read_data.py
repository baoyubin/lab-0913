from datetime import datetime

import transbigdata as tbd
import pandas as pd
import numpy as np

def read_taxi(load_map, bounds, params):
    data = pd.read_csv('./Taxi/real_time load/taxi.csv')
    time = pd.to_datetime(data['time'])
    data = data.set_index(time)
    dataset = pd.DataFrame(data.resample('1T'))
    dataset = dataset[dataset[0].dt.hour.isin(np.arange(6, 18))]
    for row in dataset.itertuples():
        a = row[2].groupby(['LONCOL', 'LATCOL'])['time'].count().reset_index()
        load_map.append(a)