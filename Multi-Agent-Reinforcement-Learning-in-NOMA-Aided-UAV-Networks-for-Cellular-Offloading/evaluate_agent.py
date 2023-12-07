#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :evaluate_agent.py
@说明        :用来评估存储的模型
@时间        :2023/11/27 16:30:24
@作者        :leo
@版本        :1.0
'''
import numpy as np
import NOMA as noma
import OMA as oma
import Kmeans as km
import copy 
from tensorflow import keras

def evaluate_OMA(name):
    model = keras.models.load_model("model/"+name)
    env = oma.SystemModel() # create an enviroomaent
    User_AS_List = km.User_association(env.PositionOfUAVs, env.PositionOfUsers,oma.NumberOfUAVs, oma.NumberOfUsers) # user association after each period because users are moving
    env.User_randomMove(oma.MAXUserspeed,oma.NumberOfUsers) # move users
    T = 60 # total time slots (steps)
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate
    env.Reset_position()
    for t in range(T):
        for UAV in range(oma.NumberOfUAVs):
            Distence_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, oma.NumberOfUAVs, oma.NumberOfUsers) # Calculate the distance for each UAV-users
            PL_for_CG = env.Get_Propergation_Loss(Distence_CG,env.PositionOfUAVs,oma.NumberOfUAVs, oma.NumberOfUsers, oma.F_c) # Calculate the pathloss for each UAV-users
            CG = env.Get_Channel_Gain_OMA(oma.NumberOfUsers, PL_for_CG, User_AS_List,oma.NoisePower) # Calculate the channel gain for each UAV-users
            Eq_CG = env.Get_Channel_Gain_OMA(oma.NumberOfUsers, PL_for_CG, User_AS_List,oma.NoisePower) # Calculate the equivalent channel gain to determine the decoding order
            State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
            action_name =np.argmax(model.predict(State)) # agent calculate action
            env.Get_Channel_Gain_OMA(action_name,UAV,User_AS_List,Eq_CG) # take action in the enviroomaent
            Distence = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, oma.NumberOfUAVs, oma.NumberOfUsers) # after taking actions, calculate the distance again
            P_L = env.Get_Propergation_Loss(Distence,env.PositionOfUAVs,oma.NumberOfUAVs, oma.NumberOfUsers, oma.F_c) #calculate the pathloss
            SINR=env.Get_SINR_OMA(oma.NumberOfUAVs,oma.NumberOfUsers,P_L,User_AS_List,Eq_CG, oma.NoisePower) # calculate SINR for users
            DataRate,SumRate,WorstuserRate = env.Calculate_Datarate(SINR, oma.NumberOfUsers, oma.Bandwidth) # calculate data rate, sum rate and the worstusers data rate
            env.User_randomMove(oma.MAXUserspeed,oma.NumberOfUsers) # move users
            if UAV==(oma.NumberOfUAVs-1):
                Rate_during_t = copy.deepcopy(SumRate)
                datarate_seq[t] = Rate_during_t
        Through_put = np.sum(datarate_seq) # calculate throughput for an episode
    print('Through_put=',Through_put)

def evaluate_NOMA(name):
    model = keras.models.load_model("model/"+name)
    env = noma.SystemModel() # create an environomaent
    User_AS_List = km.User_association(env.PositionOfUAVs, env.PositionOfUsers,noma.NumberOfUAVs, noma.NumberOfUsers) # user association after each period because users are moving
    env.User_randomMove(noma.MAXUserspeed,noma.NumberOfUsers) # move users
    T = 60 # total time slots (steps)
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate
    env.Reset_position()
    for t in range(T):
        for UAV in range(noma.NumberOfUAVs):
            Distence_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, noma.NumberOfUAVs, noma.NumberOfUsers) # Calculate the distance for each UAV-users
            PL_for_CG = env.Get_Propergation_Loss(Distence_CG,env.PositionOfUAVs,noma.NumberOfUAVs, noma.NumberOfUsers, noma.F_c) # Calculate the pathloss for each UAV-users
            CG = env.Get_Channel_Gain_NOMA(noma.NumberOfUAVs, noma.NumberOfUsers, PL_for_CG, User_AS_List,noma.NoisePower) # Calculate the channel gain for each UAV-users
            Eq_CG = env.Get_Channel_Gain_NOMA(noma.NumberOfUAVs, noma.NumberOfUsers, PL_for_CG, User_AS_List,noma.NoisePower) # Calculate the equivalent channel gain to determine the decoding order
            State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
            action_name =np.argmax(model.predict(State)) # agent calculate action
            env.take_action_NOMA(action_name,UAV,User_AS_List,Eq_CG) # take action in the environomaent
            Distence = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, noma.NumberOfUAVs, noma.NumberOfUsers) # after taking actions, calculate the distance again
            P_L = env.Get_Propergation_Loss(Distence,env.PositionOfUAVs,noma.NumberOfUAVs, noma.NumberOfUsers, noma.F_c) #calculate the pathloss
            SINR=env.Get_SINR_NNOMA(noma.NumberOfUsers,P_L,User_AS_List,Eq_CG, noma.NoisePower)
            DataRate,SumRate,WorstuserRate = env.Calculate_Datarate(SINR, noma.NumberOfUsers, noma.Bandwidth) # calculate data rate, sum rate and the worstusers data rate
            env.User_randomMove(noma.MAXUserspeed,noma.NumberOfUsers) # move users
            if UAV==(noma.NumberOfUAVs-1):
                Rate_during_t = copy.deepcopy(SumRate)
                datarate_seq[t] = Rate_during_t
        Through_put = np.sum(datarate_seq) # calculate throughput for an episode
    print('Through_put=',Through_put)

evaluate_OMA("OMA_agent")
evaluate_NOMA("NOMA_agent")
