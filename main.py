from SSFCM import *
from doc_du_lieu import *
import numpy as np
import pandas as pd
from init_semi_data import *

if __name__ == "__main__":
    import time
    from utility import round_float, extract_labels
    # from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
    from validity import dunn, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report_fcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(partition_entropy(U)),  # PE
            wdvl(Xie_Benie(X, V, U)),  # XB
        ]
        return SPLIT.join(kqdg)
    data,class_data=docDuLieu()
    # print(data)
    
    L=init_semi_data(class_data,0.3)
    print('L:', L)
    
    
    ssfcm=SSFCM(clusters=3)
    U_bar=ssfcm.init_U_bar(L)
    
    
    centroids,U,labels,i=ssfcm.fit(data,L)
    
    
    print('Ma trận thành viên : ',U)
    
    # print(np.sum(U,axis=1))
    print('Tâm cụm : ',centroids)
    print('Nhãn cụm : ',labels)
    print('Số lần lặp : ',i)

    titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'PE-', 'XB-']
    print(SPLIT.join(titles))
    print(write_report_fcm(alg='FCM', index=0, process_time=ssfcm.process_time, step=i, X=data, V=centroids, U=U))
        
    
    
    
    
    
    

   
    
    
    
    
    