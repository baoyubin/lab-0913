import transbigdata as tbd
class MapConfig:
    def __init__(self):
        self.bounds = [121.46, 31.20, 121.54, 31.28]
        self.params = tbd.area_to_params(self.bounds, accuracy=200, method='rect')