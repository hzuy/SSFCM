import numpy as np
from doc_du_lieu import *

def init_semi_data(labels, ratio):
    # ratio: Tỉ lệ nhãn
    # Chuyển đổi nhãn từ String -> int
    convert_label = np.unique(labels)
    label_to_index = {label:index for index,label in enumerate(convert_label)}
    labels = np.array([label_to_index[label] for label in labels])

    # Lấy số lượng điểm dữ liệu không làm dữ liệu bán gíam sát
    n_not_semi = int(len(labels) * (1-ratio))

    # Chọn ra n_semi điểm ngẫu nhiên chuyển label 
    np.random.seed(42)
    unlabel = np.random.choice(a=[i for i in range(len(labels))], size=n_not_semi,replace=False)

    # Chuyển các giá trị thành -1
    for index in range(len(labels)):
        if index in unlabel:
            labels[index] = -1

    return labels
