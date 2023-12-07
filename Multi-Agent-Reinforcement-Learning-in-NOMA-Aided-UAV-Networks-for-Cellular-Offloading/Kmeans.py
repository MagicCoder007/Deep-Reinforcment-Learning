#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :Kmeans.py
@说明        :使用kmeans算法来完成用户关联
@时间        :2023/11/27 16:38:01
@作者        :leo
@版本        :1.0
'''
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings

def User_association(UAV_Position,User_Position,UAVsnumber, Usersnumber):
    """
    用户关联

    Args:
        UAV_Position (array): 三维UAV的位置
        User_Position (array): 用户位置的三维位置
        UAVsnumber (int): UAV的数量
        Usersnumber (int): 用户的数量

    Returns:
        df array: 用户管理数组
    """
    # this function is for user association
    User_Position_array = np.zeros([Usersnumber,2])
    # convert data type
    # User_Position是df，因此需要转换成np
    User_Position_array[:, 0] = User_Position.iloc[0,:].T
    User_Position_array[:, 1] = User_Position.iloc[1,:].T
    # K-means approach for user association 根据用户位置聚类
    K_means_association = KMeans(n_clusters=UAVsnumber).fit(User_Position_array)
    User_cluster = K_means_association.labels_# 每个用户归属哪个类别（分出三个类）
    Cluster_center = K_means_association.cluster_centers_
    # check if the clusters are with equal users
    # 每个cluster只能有相同的用户数
    for dectecter in range(UAVsnumber):
        user_numberincluster=np.where(User_cluster == dectecter)[0]
        if len(user_numberincluster) == (Usersnumber/UAVsnumber):
                pass
        else:
            cluster_redun = []
            cluster_lack = []
            Cluster_center_of_lack=[]
            for ck_i in range(len(Cluster_center)): # Find clusters with more or less elements
                User_for_cluster_i = np.where(User_cluster==ck_i)
                if np.size(User_for_cluster_i) > (Usersnumber/UAVsnumber): # Find clusters with redundant elements
                    for i in range(int(np.size(User_for_cluster_i)-(Usersnumber/UAVsnumber))):
                        cluster_redun.append(ck_i)
                if np.size(User_for_cluster_i) < (Usersnumber/UAVsnumber): # Find clusters short elements
                    for i in range(int((Usersnumber/UAVsnumber)-np.size(User_for_cluster_i))):
                        cluster_lack.append(ck_i)
                        Cluster_center_of_lack.append(Cluster_center[ck_i, :])
            # Assign redundant users to the cluster short users
            for fixer_i in range(np.size(cluster_lack)):
                cluster_lack_fixing = cluster_lack[fixer_i]
                Lacker_Center = Cluster_center_of_lack[fixer_i]
                Redun_cluster = cluster_redun[fixer_i]
                Redun_cluster_user = np.where(User_cluster==Redun_cluster) # find redundant users
                Redun_cluster_user_postion = User_Position_array[Redun_cluster_user,:] # find redundant users' position
                distence_U2C = np.zeros(np.size(Redun_cluster_user)) # Find the closest to the few user groups
                for find_i in range(np.size(Redun_cluster_user)):
                    distence_U2C[find_i] = np.linalg.norm(Redun_cluster_user_postion[:,find_i]-Lacker_Center)
                min_distence_user_order = np.where(distence_U2C==np.min(distence_U2C))
                Redun_cluster_user_list = Redun_cluster_user[0] # Data type conversion
                Min_d_User_idx = Redun_cluster_user_list[int(min_distence_user_order[0])]
                User_cluster_fixed = User_cluster
                User_cluster_fixed[Min_d_User_idx] = cluster_lack_fixing
            User_cluster = User_cluster_fixed
    if sum(User_cluster) != (UAVsnumber - 1) * Usersnumber / 2:
        warnings.warn("User association wrong")
    # Choose the nearest UAV for each user clusters
    UAV_Position_array = np.zeros([UAVsnumber, 2])
    UAV_Position_array[:, 0] = UAV_Position.iloc[0, :].T
    UAV_Position_array[:, 1] = UAV_Position.iloc[1, :].T
    # data frame for saving user association indicators
    # 转化为df格式
    User_association_list = pd.DataFrame(
        np.zeros((1, Usersnumber)),
        columns=np.arange(Usersnumber).tolist(),
    )  
    #将cluster和UAV相关联
    for UAV_name in range(UAVsnumber):
        distence_UAVi2C = np.zeros(UAVsnumber)
        for cluster_center_i in range(UAVsnumber):
            distence_UAVi2C[cluster_center_i] = np.linalg.norm(UAV_Position_array[UAV_name,: ] - Cluster_center[cluster_center_i])
        Servied_cluster = np.where(distence_UAVi2C==np.min(distence_UAVi2C)) # assciate UAV_name with the closest cluster
        Cluster_center[Servied_cluster] = 9999  # remove the selected cluster
        Servied_cluster_list = Servied_cluster[0]
        Servied_users = np.where(User_cluster==Servied_cluster_list)
        Servied_users_list = Servied_users[0]
        for i in range(np.size(Servied_users)):
            User_association_list.iloc[0,Servied_users_list[i]] = int(UAV_name) # fill UAV names in User_association_list
        User_association_list = User_association_list.astype('int') #converted data type to int
    
    print(User_association_list.values)

    return User_association_list