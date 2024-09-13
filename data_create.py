import Taxi.demo_create as taxi
import Bus.demo_create as bus
import transbigdata as tbd
if __name__ == '__main__':
    # bounds = [121.4, 31.15, 121.6, 31.35]
    bounds = [121.46, 31.20, 121.56, 31.30]
    params = tbd.area_to_params(bounds, accuracy=200, method='rect')
    # taxi.show_taxidata(bounds, params)
    # taxi.create_taxidata(bounds, params)
    print("taxi END")
    bus.show_data(bounds,params)
    # bus.create_busdata(bounds, params)
    ##xiugaishijian
    print("bus END")