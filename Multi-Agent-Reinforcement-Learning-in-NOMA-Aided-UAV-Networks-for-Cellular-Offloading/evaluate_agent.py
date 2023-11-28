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
import NOMA as nm
import Kmeans as km
import copy 
from tensorflow import keras

def evaluate(name):
    model = keras.models.load_model("model/"+name)
    env = nm.SystemModel() # create an environment
    User_AS_List = km.User_association(env.PositionOfUAVs, env.PositionOfUsers,nm.NumberOfUAVs, nm.NumberOfUsers) # user association after each period because users are moving
    env.User_randomMove(nm.MAXUserspeed,nm.NumberOfUsers) # move users
    T = 60 # total time slots (steps)
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate
    env.Reset_position()
    for t in range(T):
        for UAV in range(nm.NumberOfUAVs):

            Distence_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, nm.NumberOfUAVs, nm.NumberOfUsers) # Calculate the distance for each UAV-users
            PL_for_CG = env.Get_Propergation_Loss(Distence_CG,env.PositionOfUAVs,nm.NumberOfUAVs, nm.NumberOfUsers, nm.F_c) # Calculate the pathloss for each UAV-users
            CG = env.Get_Channel_Gain_NOMA(nm.NumberOfUAVs, nm.NumberOfUsers, PL_for_CG, User_AS_List,nm.NoisePower) # Calculate the channel gain for each UAV-users
            Eq_CG = env.Get_Channel_Gain_NOMA(nm.NumberOfUAVs, nm.NumberOfUsers, PL_for_CG, User_AS_List,nm.NoisePower) # Calculate the equivalent channel gain to determine the decoding order

            State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
            action_name =np.argmax(model.predict(State)) # agent calculate action
            env.take_action_NOMA(action_name,UAV,User_AS_List,Eq_CG) # take action in the environment

            Distence = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, nm.NumberOfUAVs, nm.NumberOfUsers) # after taking actions, calculate the distance again
            P_L = env.Get_Propergation_Loss(Distence,env.PositionOfUAVs,nm.NumberOfUAVs, nm.NumberOfUsers, nm.F_c) #calculate the pathloss
            SINR=env.Get_SINR_NNOMA(nm.NumberOfUAVs,nm.NumberOfUsers,P_L,User_AS_List,Eq_CG, nm.NoisePower) # calculate SINR for users
            DataRate,SumRate,WorstuserRate = env.Calcullate_Datarate(SINR, nm.NumberOfUsers, nm.Bandwidth) # calculate data rate, sum rate and the worstusers data rate
            env.User_randomMove(nm.MAXUserspeed,nm.NumberOfUsers) # move users

            # print('UE',env.PositionOfUsers) #check user position
            # print('UAV',env.PositionOfUAVs) #check UAV position

            # save data after all UAVs moved
            if UAV==(nm.NumberOfUAVs-1):
                Rate_during_t = copy.deepcopy(SumRate)
                datarate_seq[t] = Rate_during_t

        Through_put = np.sum(datarate_seq) # calculate throughput for an episode
    print('Through_put=',Through_put)
    
evaluate("init")
evaluate("agent")