from FCM import *
from scipy.spatial.distance import cdist
import numpy as np

class SSFCM(FCM):
    def __init__(self, clusters, m=2, max_iters=1000, eps=1e-5,per=30):
        super().__init__(clusters, m, max_iters, eps)
        self.U_bar=None
        self.per=per
    
    #khởi tạo ma trận U ngang
    def init_U_bar(self,labels):
         # Xác định các nhãn duy nhất và chuyển nhãn về dạng số 
        unique_classes, numeric_labels = np.unique(labels, return_inverse=True)

        # Khởi tạo ma trận U_bar có kích thước [N x C] với tất cả giá trị ban đầu là 0
        num_samples = len(numeric_labels)
        self.U_bar = np.zeros((num_samples, self.clusters))
        
            # Xác định số lượng mẫu được gán nhãn cứng (dựa trên self.per%)
        num_labeled_samples = int((num_samples * self.per) / 100)

         # Chọn ngẫu nhiên các chỉ số của dữ liệu được gán nhãn cứng
        labeled_indices = np.random.choice(num_samples,size=num_labeled_samples,replace=False)  # Đảm bảo các chỉ số không trùng lặp)

        # Gán nhãn cứng vào các vị trí được chọn
        for idx in labeled_indices:
            cluster_label = labels[idx]  # Lấy nhãn cụm của mẫu
            self.U_bar[idx, cluster_label] = 1    # Gán nhãn cứng tại vị trí tương ứng

        return self.U_bar
    
    #cập nhật tâm cụm
    def calculate_centroid(self,data,U):
        U_U_bar=U-self.U_bar
        return super().calculate_centroid(data,U_U_bar)
    
    #cập nhật ma trận thành viên
    def update_membership_matrix(self,data,centroid):
        u=super().update_membership_matrix(data,centroid)
        matrix_1_Sum_U=1-np.sum(self.U_bar,axis=1,keepdims=True)
        return self.U_bar+matrix_1_Sum_U*u
    
    def fit(self, data,labels):
        self.U_bar=self.init_U_bar(labels)
        return super().fit(data)
    
    
    

    
        
    
        
        
        
    
    
    
    
      
    