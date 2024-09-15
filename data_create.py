from transbigdata import sample_duration

import Taxi.demo_create as taxi
import Bus.demo_create as bus
from MapConfig import MapConfig
import transbigdata as tbd
if __name__ == '__main__':
    # bounds = [121.4, 31.15, 121.6, 31.35]
    config = MapConfig()
    ##修改bounds/params 请先运行data_create
    bounds = config.bounds
    params = config.params
    # taxi.show_taxidata(bounds, params)
    taxi.create_taxidata(bounds, params)
    print("taxi END")
    # bus.show_data(bounds, params)
    # bus.create_busdata(bounds, params)
    ##xiugaishijian
    print("bus END")


