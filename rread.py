import pandas as pd
import transbigdata as tbd
import numpy as np
from matplotlib import pyplot as plt

from my_map import Bus_map, Load_map
from my_env_back2 import Env
import math
from pymoo.algorithms.moo.nsga2 import NSGA2


if __name__ == '__main__':
    with open('17_34_12_rw.txt', 'r') as file:
        # 使用列表推导式读取所有行并去除换行符，然后加入列表
        lines = [float(line.strip()) for line in file]