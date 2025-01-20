import pandas as pd
import numpy as np
from init_semi_data import *

def docDuLieu():
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv("dataset/Iris/data.csv")
    class_data=data['class']
    data = data.drop(columns='class')
    data=data.values
    return data,class_data

