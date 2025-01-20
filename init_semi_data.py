import numpy as np
from doc_du_lieu import *

def  init_semi_data(labels, ratio):
    # ratio: Tỉ lệ nhãn
    # Chuyển đổi nhãn từ String -> int
    convert_label = np.unique(labels)
    label_to_index = {label:index for index,label in enumerate(convert_label)}
    labels = np.array([label_to_index[label] for label in labels])

    # Lấy số lượng điểm dữ liệu  làm dữ liệu bán gíam sát
    n_semi = int(len(labels) * (ratio))

    # Chọn ra n_semi điểm ngẫu nhiên chuyển label 
    np.random.seed(42)
    label = np.random.choice(a=len(labels), size=n_semi,replace=False)

    # Chuyển các giá trị thành -1
    for index in range(len(labels)):
        if index not in label:
            labels[index] = -1

    return labels
