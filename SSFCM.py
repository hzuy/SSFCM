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
        #khởi tạo ma trận 0 có kích thước NxC
        self.U_bar=np.zeros((len(labels),self.clusters))
        #phần tử có nhãn thì gán giá trị 1
        for i in range(len(labels)):
            if labels[i]!=-1:
                self.U_bar[i][labels[i]]=1
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
    
    
    

    
        
    
        
        
        
    
    
    
    
      
    