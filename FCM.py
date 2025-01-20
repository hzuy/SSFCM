import numpy as np
from scipy.spatial.distance import cdist
class FCM :
    
    def __init__(self,clusters,m=2,max_iters=1000,eps=1e-5):
        self.clusters=clusters
        self.m=m
        self.max_iters=max_iters
        self.eps=eps
        self.process_time=0
    
    
    
    # khởi tạo ma trận thành viên
    def initialize_ramdom_membership_matrix(self,data):
        np.random.seed(seed=42)
        #lấy số lượng điểm dữ liệu cần khởi tạo ma trận thành viên U
        n=data.shape[0]
        #khởi tạo ma trận ngẫu nhiên kích thước n x c
        U=np.random.rand(n,self.clusters)
         #chuẩn hóa để tổng các cột là 1
        U=U/np.sum(U,axis=1,keepdims=True) 
        return U
    
    #tính toán tâm cụm dựa theo ma trận thành viên U
    def calculate_centroid(self,data,U):
         # Nâng ma trận thành viên U lên U mũ m
        Um = U ** self.m
        TS = np.dot(Um.T , data) 
        MS = np.sum(Um.T, axis=1,keepdims=True)
        return TS /MS  

    
    
    #cập nhật ma trận thành viên dựa theo khoảng cách đến tâm cụm
    def update_membership_matrix(self,data,centroid):
        #số cụm
        c=self.clusters
        #tính ma trận khoảng cách từ từng điểm đến từng tâm cụm
        D=[]
        distance = cdist(data,centroid,metric='euclidean') **(2/(self.m-1))
        for j in range(c):
            distance_ij=distance[:,j]   #data:[n,d]    #[n,1]
            # print(distance_ij)
            D.append(distance_ij)  #[c,d]
        TS=1/np.array(D)
        MS=np.sum(TS,axis=0)
        # print(TS.shape)
        U=TS/MS
        U=np.squeeze(U).T
        
        return U
    
        
        
        
        
    
    def fit(self,data):
        #khởi tạo ma trận thành viên
        U=self.initialize_ramdom_membership_matrix(data)
        # print(U)
        for i in  range(self.max_iters):
            #tính toán tâm cụm
            centroids=self.calculate_centroid(data,U)
            #cập nhật ma trận thành viên
            new_U=self.update_membership_matrix(data,centroids)
            if (np.linalg.norm(new_U-U))<self.eps:
                break
            U=np.copy(new_U) #cập nhật cho các lần lặp sau
        #dự đoán các cụm của từng điểm dữ liệu
        labels=np.argmax(new_U,axis=1)
        return centroids,U,labels,i
    
    
     #tính label theo U
    def calcutlate_labels(self,U):
        labels=np.argmax(U, axis=1)
        return labels
   
    # n=data.shape[0]
        # clusters=self.clusters
        # #tạo ma trận thành viên mới có kích thước n x clusters
        # U=np.zeros((n,clusters))
        # for i in range (n):
        #     for j in range (clusters):
        #         #tính khoảng cách điểm i dến tâm cụm j
        #         distance_ij=np.linalg.norm(data[i]-centroid[j])
                
        #         #tính tổng nghịch đảo khoảng cách cho công thức mờ
        #         mau_so=0
        #         for k in range (clusters):
        #              distance_ik=np.linalg.norm(data[i]-centroid[k])
        #              mau_so=mau_so+(distance_ij/distance_ik)**(self.m-1)
        #         #cập nhật giá trị ma trận thành viên
        #         U[i,j]=1/mau_so
       
        # return U
        
    
        
        
        