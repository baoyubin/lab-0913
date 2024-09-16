import transbigdata as tbd
class MapConfig:
    def __init__(self):
        self.bounds = [121.46, 31.20, 121.54, 31.28]
        self.gripAc = 200
        self.params = tbd.area_to_params(self.bounds, accuracy=self.gripAc, method='rect')