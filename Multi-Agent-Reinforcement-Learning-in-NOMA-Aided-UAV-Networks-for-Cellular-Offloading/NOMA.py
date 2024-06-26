#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :NOMA.py
@说明        :NOMA方式
@时间        :2024/05/27 20:27:49
@修改        :leo
@版本        :1.0
'''

import numpy as np
import math
import pandas as pd
import warnings
import copy
import matplotlib.pyplot as plt
import user_association as ua
from DQN import DQN
import pathlib


# ********************************** #
# 
# ********************************** #

# ================================ #
# 定义一些常数
#random seed control if necessary
#np.random.seed(1)
# 当前的目录
path = pathlib.Path(__file__).parent.resolve()
figure_path = path.joinpath("figure")
model_path = path.joinpath("model")
data_path = path.joinpath("data")

# set up service area range
ServiceZone_X = 500
ServiceZone_Y = 500
Hight_limit_Z = 150

# set up users' speed
MAXUserspeed = 0.5 #m/s
UAV_Speed = 5 #m/s

UserNumberPerCell = 2 # user number per UAV
NumberOfUAVs = 3 # number of UAVs
NumberOfCells = NumberOfUAVs # Each UAV is responsible for one cell
NumberOfUsers = NumberOfUAVs * UserNumberPerCell
F_c = 2 # carrier frequency/GHz
# NOMA比OMA的带宽翻倍了
Bandwidth = 30 #khz
R_require = 0.1 # QoS data rate requirement kb
Power_level= 3 # Since DQN can only solve discrete action spaces, we set several discrete power gears, Please note that the change of power leveal will require a reset on the action space

# 仅仅是为了数值可计算，因此功率和噪声都乘以一个常数
amplification_constant = 10000 # Since the original power and noise values are sometims negligible, it may cause NAN data. We perform unified amplification to avoid data type errors
UAV_power_unit = 100 * amplification_constant # 100mW=20dBm
NoisePower = 10**(-9) * amplification_constant # noise power
# ================================ #

class SystemModel(object):
    """
    系统模型相关的属性和函数封装在类中

    Args:
        object (_type_): _description_
    """
    def __init__(
            self,
    ):
        # Initialize area
        self.Zone_border_X = ServiceZone_X
        self.Zone_border_Y = ServiceZone_Y
        self.Zone_border_Z = Hight_limit_Z
        # Initialize UAV and their location
        self.UAVspeed = UAV_Speed
        self.UAV_number = NumberOfUAVs
        self.UserperCell = UserNumberPerCell
        self.U_idx = np.arange(NumberOfUAVs) # set up serial number for UAVs
        self.PositionOfUAVs = pd.DataFrame(
            np.zeros((3,NumberOfUAVs)),
            columns=self.U_idx.tolist(),    # Data frame for saving UAVs' position
        )
        self.PositionOfUAVs.iloc[0, :] = [100, 200, 400]  # UAVs' initial x
        self.PositionOfUAVs.iloc[1, :] = [100, 400, 200]  # UAVs' initial y
        self.PositionOfUAVs.iloc[2, :] = [100, 100, 100]  # UAVs' initial z

        # Initialize users and users' location
        self.User_number = NumberOfUsers
        self.K_idx = np.arange(NumberOfUsers) # set up serial number for users
        self.PositionOfUsers = pd.DataFrame(
            np.random.random((3,NumberOfUsers)),
            columns=self.K_idx.tolist(),    # Data frame for saving users' position
        )
        # self.PositionOfUsers.iloc[0,:] = [204.91, 493.51, 379.41, 493.46, 258.97, 53.33] # users' initial x
        # self.PositionOfUsers.iloc[1, :] = [219.75, 220.10, 49.81, 118.10, 332.59, 183.11] # users' initial y
        self.PositionOfUsers.iloc[0,:] =  np.random.randint(0,500,6)
        self.PositionOfUsers.iloc[1, :] = np.random.randint(0,500,6)
        self.PositionOfUsers.iloc[2, :] = 0 # users' hight is assumed to be 0

        # record initial state
        self.Init_PositionOfUsers = copy.deepcopy(self.PositionOfUsers)
        self.Init_PositionOfUAVs = copy.deepcopy(self.PositionOfUAVs)

        # initialize a array to store state
        self.State = np.zeros([1, NumberOfUAVs * 3 + NumberOfUsers ], dtype=float)

        # Create a data frame for storing transmit power
        self.Power_allocation_list = pd.DataFrame(
            np.ones((1, NumberOfUsers)),
            columns=np.arange(NumberOfUsers).tolist(),
        )
        self.Power_unit = UAV_power_unit
        self.Power_allocation_list = self.Power_allocation_list * self.Power_unit

        # data frame to save distance
        self.Distance = pd.DataFrame(
            np.zeros((self.UAV_number, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # data frame to save pathloss
        self.Propergation_Loss = pd.DataFrame(
            np.zeros((self.UAV_number, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # create a data frame to save channel gain
        self.ChannelGain_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # create a data frame to save equivalent channel gain
        self.Eq_CG_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # Create a data frame to save SINR
        self.SINR_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # Create a data frame to save datarate
        self.Datarate = pd.DataFrame(
        np.zeros((1, self.User_number)),
        columns=np.arange(self.User_number).tolist(),)

        # amplification_constant as mentioned above
        self.amplification_constant = amplification_constant


    def User_randomMove(self,MAXspeed,NumberofUsers):
        """
        用户随机移动

        Args:
            MAXspeed (float): 最大速率
            NumberofUsers (int): 用户数量
        """
        self.PositionOfUsers.iloc[[0,1],:] += np.random.randn(2,NumberofUsers)*MAXspeed # users random move
        return


    def Get_Distance_U2K(self,UAV_Position,User_Position,UAVsnumber,Usersnumber):
        """
        计算UAV和用户之间的距离
        this function is for calculating the distance between users and UAVs
        Args:
            UAV_Position (array): UAV位置
            User_Position (array): 用户位置
            UAVsnumber (int): UAV的数量
            Usersnumber (int): 用户数量
        Returns:
            array: 计算完的距离
        """
        for i in range(UAVsnumber):
            for j in range(Usersnumber):
                self.Distance.iloc[i,j] = np.linalg.norm(UAV_Position.iloc[:,i]-User_Position.iloc[:,j]) # calculate Distance betwen UAV i and User j

        return self.Distance


    def Get_Propergation_Loss(self,distance_U2K,UAV_Position,UAVsnumber,Usersnumber,f_c):
        """
        参考Propagation Model
        this function is for calculating the pathloss between users and UAVs
        Calculate average loss for each user,  this pathloss model is for 22.5m<h<300m d(2d)<4km
        Args:
            UAV_Position (array): UAV位置
            User_Position (array): 用户位置
            UAVsnumber (int): UAV的数量
            Usersnumber (int): 用户数量
            f_c (int): 使用的频率
        Returns:
            array: 传播损耗
        """
        for i in range(UAVsnumber):# Calculate average loss for each user,  this pathloss model is for 22.5m<h<300m d(2d)<4km
            for j in range(Usersnumber):
                UAV_Hight=UAV_Position.iloc[2,i]
                D_H = np.sqrt(np.square(distance_U2K.iloc[i,j])-np.square(UAV_Hight)) # calculate distance
                # calculate the possibility of LOS/NLOS
                d_0 = np.max([(294.05*math.log(UAV_Hight,10)-432.94),18])
                p_1 = 233.98 * math.log(UAV_Hight,10) - 0.95
                if D_H <= d_0:
                    P_Los = 1.0
                else:
                    P_Los = d_0/D_H + math.exp(-(D_H/p_1)*(1-(d_0/D_H)))
                if P_Los>1:
                    P_Los = 1
                P_NLos = 1 - P_Los
                # calculate the passloss for LOS/NOLS
                L_Los = 30.9 + (22.25-0.5*math.log(UAV_Hight,10))*math.log(distance_U2K.iloc[i,j],10) + 20*math.log(f_c,10)
                L_NLos = np.max([L_Los,32.4+(43.2-7.6*math.log(UAV_Hight,10))*math.log(distance_U2K.iloc[i,j],10)+20*math.log(f_c,10)])
                Avg_Los = P_Los*L_Los + P_NLos*L_NLos # average pathloss
                gain = np.random.rayleigh(scale=1, size=None)*pow(10,(-Avg_Los/10)) # random fading
                self.Propergation_Loss.iloc[i,j] = gain #save pathloss
        return self.Propergation_Loss


    def Get_Channel_Gain_NOMA(self,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power):
        """
        计算每个用户的SINR值
        This function is to calculate the SINR for every users

        Args:
            Usersnumber (int): 用户数量
            PropergationLosslist (array): 传播损失数组
            UserAssociationlist (array): 用户关联数组
            Noise_Power (float): 噪声功率
        Returns:
            array: 信道增益数组
        """
        # this function is for calculating channel gain
        for j in range(Usersnumber):  # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = self.amplification_constant * PropergationLosslist.iloc[i_Server_UAV, j]
            ChannelGain = Signal_power / ( Noise_Power) # calculate channel gain
            self.ChannelGain_list.iloc[0, j] = ChannelGain # save channel gain
        return self.ChannelGain_list

    def Get_SINR_NNOMA(self,Usersnumber,PropergationLosslist,UserAssociationlist,ChannelGain_list,Noise_Power):
        """
        获取所有用户的SINR数组（NOMA方式）

        Args:
            Usersnumber (int): 用户的数量
            PropergationLosslist (array): 传播损失数组
            UserAssociationlist (array): 用户关联数组
            ChannelGain_list (array): 信道增益数组
            Noise_Power (float): 噪声功率
        Returns:
            array: SINR数组
        """
        #This function is to calculate the SINR for every users
        for j in range(Usersnumber): # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service 'j_idx' represents the other users
            i_Server_UAV = UserAssociationlist.iloc[0,j]
            Signal_power = self.Power_allocation_list.iloc[0,j] * PropergationLosslist.iloc[i_Server_UAV,j] # read the sinal power from power allocation list
            I_inter_cluster = 0
            for j_idx in range(Usersnumber): # calculate Interference for user j
                if UserAssociationlist.iloc[0,j_idx] == i_Server_UAV:
                    if ChannelGain_list.iloc[0,j] < ChannelGain_list.iloc[0,j_idx] and j!=j_idx: #find 'stronger' users in same cluster to count intra cluster interference
                        I_inter_cluster = I_inter_cluster + (
                                    self.Power_allocation_list.iloc[0, j_idx] * PropergationLosslist.iloc[
                                i_Server_UAV, j])  #calculate intra cluster interference
                else:
                    Inter_UAV = UserAssociationlist.iloc[0,j_idx] # calculate inter cluster interference from other UAVs
                    I_inter_cluster = I_inter_cluster + (self.Power_allocation_list.iloc[0,j_idx] * PropergationLosslist.iloc[Inter_UAV,j])#
            SINR = Signal_power/(I_inter_cluster + Noise_Power) # calculate SINR and save it
            self.SINR_list.iloc[0,j] = SINR

        return self.SINR_list


    def Calculate_Datarate(self,SINRlist,Usersnumber,B): # calculate data rate for all users
        """
        计算速率        
        Args:
            SINRlist (array): SINR数组
            Usersnumber (int): 用户数量
            B (int): 带宽

        Returns:
            array:每个用户的速率 
            float:速率和
            float:最差速率
        """
        for j in range(Usersnumber):
            if SINRlist.iloc[0,j] <=0:
                warnings.warn('SINR wrong') # A data type error may occur when the data rate is too small, thus we set up this alarm
            self.Datarate.iloc[0,j] = B*math.log((1+SINRlist.iloc[0,j]),2)
        SumDataRate = sum(self.Datarate.iloc[0,:])
        Worst_user_rate = min(self.Datarate.iloc[0,:])
        return self.Datarate,SumDataRate,Worst_user_rate


    def Reset_position(self): # save initial state for environment reset
        """
        用户和UAV设置成初始位置
        """
        self.PositionOfUsers = copy.deepcopy(self.Init_PositionOfUsers)
        self.PositionOfUAVs = copy.deepcopy(self.Init_PositionOfUAVs)
        return


    def Create_state_Noposition(self,serving_UAV,User_association_list,User_Channel_Gain):
        """
        创建状态
        状态包括UAV的位置和CSI信息

        Args:
            serving_UAV (int): 选择的UAV
            User_association_list (array): 用户关联数组
            User_Channel_Gain (array): 信道增益数组

        Returns:
            df: 状态
        """
        # Create state, pay attention we need to ensure UAVs and users who are making decisions always input at the fixed neural node to achieve MDQN
        UAV_position_copy = copy.deepcopy(self.PositionOfUAVs.values)
        UAV_position_copy[:,[0,serving_UAV]] = UAV_position_copy[:,[serving_UAV,0]] # adjust the input node of serving UAV to ensure it is fixed
        User_Channel_Gain_copy = copy.deepcopy(User_Channel_Gain.values[0])
        # save UAV positions as a part of the state
        for UAV in range(NumberOfUAVs):
            self.State[0, 3 * UAV:3 * UAV + 3] = UAV_position_copy[:, UAV].T 

        User_association_copy = copy.deepcopy(User_association_list.values)
        desirable_user = np.where(User_association_copy[0]==serving_UAV)[0] # find out the current served users

        for i in range(len(desirable_user)):
             User_Channel_Gain_copy[i],User_Channel_Gain_copy[desirable_user[i]] = User_Channel_Gain_copy[desirable_user[i]],User_Channel_Gain_copy[i] # Similarly, adjust the input node of the current served users
        # save CSI of users in state
        for User in range(NumberOfUsers):
            self.State[0,(3*UAV+3)+User] = User_Channel_Gain_copy[User].T 

        Stat_for_return = copy.deepcopy(self.State)
        return Stat_for_return


    def take_action_NOMA(self,action_number,acting_UAV,User_asso_list,ChannelGain_list):
        """
        执行Agent的动作
        相当于将动作拆分成了UAV移动和功率分配

        Args:
            action_number (int): 当前agent选择的动作
            acting_UAV (int): 当前执行的UAV
            User_asso_list (_type_): 用户关联数组
            ChannelGain_list (_type_): 信道增益数组
        """
        UAV_move_direction = action_number % 7  #UAV has seven positional actions
        if UAV_move_direction == 0:# UAV moves along the positive half axis of the x-axis
            self.PositionOfUAVs.iloc[0,acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[0,acting_UAV] > self.Zone_border_X:
                self.PositionOfUAVs.iloc[0, acting_UAV] = self.Zone_border_X
        elif UAV_move_direction == 1: # UAV moves along the negative half axis of the x-axis
            self.PositionOfUAVs.iloc[0, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[0, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[0, acting_UAV] = 0
        elif UAV_move_direction == 2: # UAV moves along the positive half axis of the y-axis
            self.PositionOfUAVs.iloc[1, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] > self.Zone_border_Y:
                self.PositionOfUAVs.iloc[1, acting_UAV] = self.Zone_border_Y
        elif UAV_move_direction == 3: # UAV moves along the negative half axis of the y-axis
            self.PositionOfUAVs.iloc[1, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[1, acting_UAV] = 0
        elif UAV_move_direction == 4: # UAV moves along the positive half axis of the z-axis
            self.PositionOfUAVs.iloc[2, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] > self.Zone_border_Z:
                self.PositionOfUAVs.iloc[2, acting_UAV] = self.Zone_border_Z
        elif UAV_move_direction == 5: # UAV moves along the negative half axis of the z-axis
            self.PositionOfUAVs.iloc[2, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] < 20:
                self.PositionOfUAVs.iloc[2, acting_UAV] = 20
        elif UAV_move_direction == 6: # UAV hold the position
            pass

        # Power allocation part - NOMA
        power_allocation_scheme = action_number//7  # decode the power allocation action,
        acting_user_list = np.where(User_asso_list.iloc[0,:] == acting_UAV)[0]
        First_user = acting_user_list[0]
        Second_user = acting_user_list[1]

        # SIC decoding order
        first_user_CG = ChannelGain_list.iloc[0,First_user]
        second_user_CG = ChannelGain_list.iloc[0,Second_user]
        if first_user_CG >= second_user_CG:
            User0 = Second_user
            User1 = First_user
        else:
            User0 = First_user
            User1 = Second_user

        # three power levels for each user
        # for the weak user, the power levels can be 2, 4, 7 * power unit
        if power_allocation_scheme % 3 == 0:
            self.Power_allocation_list.iloc[0,User0] = self.Power_unit*2
        elif power_allocation_scheme % 3 == 1:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*4
        elif power_allocation_scheme % 3 == 2:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*7
        # for the strong user, the power levels can be 1, 1/2, 1/4 * power unit
        if power_allocation_scheme // 3 == 0:
            self.Power_allocation_list.iloc[0,User1] = self.Power_unit
        elif power_allocation_scheme // 3 == 1:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/2
        elif power_allocation_scheme // 3 == 2:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/4


def main():
    Episodes_number = 150 # total episodes number
    Test_episodes_number = 30  # number of test episodes
    T = 60 #total time slots (steps)
    # 因为200 > T，所以T_AS = [0]
    T_AS = np.arange(0, T, 200) # time solt of user association, current setting indicate 1
    # T_AS = np.arange(0, T, 0.25) # time solt of user association, current setting indicate 1
    print(T_AS)
    env = SystemModel() # create an environment
    agent = DQN(UserNumberPerCell,NumberOfUAVs,NumberOfUsers) # crate an agent
    Epsilon = 0.9
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate
    WorstuserRate_seq = np.zeros(T) # Initialize memory to store data rate of the worst user
    Through_put_seq = np.zeros(Episodes_number) # Initialize memory to store throughput
    Worstuser_TP_seq = np.zeros(Episodes_number) # Initialize memory to store throughput of the worst user
    #保存初始化的模型
    agent.save(model_path.joinpath("NOMA_init"))
    for episode in range(Episodes_number):
        env.Reset_position()
        #if Epsilon > 0.05: # determine the minimum Epsilon value
        Epsilon -= 0.9 / (Episodes_number - Test_episodes_number) # decaying epsilon
        punishment_counter = 0 # punishment counter
        for t in range(T):
            if t in T_AS:
                # 获得用户关联，实际只执行了一次
                User_AS_List = ua.kmeans(env.PositionOfUAVs, env.PositionOfUsers,NumberOfUAVs, NumberOfUsers) # user association after each period because users are moving
            for UAV in range(NumberOfUAVs):
                # 1.计算距离，计算信道衰落，计算信道增益，计算状态，Agent预测动作，执行动作
                Distance_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, NumberOfUsers) # Calculate the distance for each UAV-users
                PL_for_CG = env.Get_Propergation_Loss(Distance_CG,env.PositionOfUAVs,NumberOfUAVs, NumberOfUsers, F_c) # Calculate the pathloss for each UAV-users
                CG = env.Get_Channel_Gain_NOMA(NumberOfUsers, PL_for_CG, User_AS_List,NoisePower) # Calculate the channel gain for each UAV-users
                State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
                action_name = agent.Choose_action(State,Epsilon) # agent calculate action
                env.take_action_NOMA(action_name,UAV,User_AS_List,CG) # take action in the environment
                # 2.计算执行动作之后的情况
                Distance = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, NumberOfUsers) # after taking actions, calculate the distance again
                P_L = env.Get_Propergation_Loss(Distance,env.PositionOfUAVs,NumberOfUAVs, NumberOfUsers, F_c) #calculate the pathloss
                SINR = env.Get_SINR_NNOMA(NumberOfUsers,P_L,User_AS_List,CG,NoisePower) # calculate SINR for users
                DataRate,SumRate,WorstuserRate = env.Calculate_Datarate(SINR, NumberOfUsers, Bandwidth) # calculate data rate, sum rate and the worstusers data rate
                # calculate reward based on sum rate and check if users meet the QoS requirement
                # 3.计算Reward
                Reward = SumRate
                if WorstuserRate < R_require:
                    #Reward = Reward/2
                    Reward = -1
                    punishment_counter+=1
                # 4.下一个状态情况
                CG_next = env.Get_Channel_Gain_NOMA(NumberOfUsers, P_L, User_AS_List,NoisePower)  # Calculate the equivalent channel gain for S_{t+1}
                Next_state = env.Create_state_Noposition(UAV,User_AS_List,CG_next) # Generate S_{t+1}
                # 5.组织经验
                #copy data for (S_t,A_t,S_t+1,R_t)
                State_for_memory = copy.deepcopy(State[0])
                Action_for_memory = copy.deepcopy(action_name)
                Next_state_for_memory = copy.deepcopy(Next_state[0])
                Reward_for_memory = copy.deepcopy(Reward)
                # 存储并训练
                agent.remember(State_for_memory, Action_for_memory, Next_state_for_memory, Reward_for_memory) #save the MDP transitions as (S_t,A_t,S_t+1,R_t)
                agent.train() # train the DQN agent
                # 移动用户
                env.User_randomMove(MAXUserspeed,NumberOfUsers) # move users
                # save data after all UAVs moved
                if UAV==(NumberOfUAVs-1):
                    Rate_during_t = copy.deepcopy(SumRate)
                    datarate_seq[t] = Rate_during_t
                    WorstuserRate_seq[t] = WorstuserRate
        Through_put = np.sum(datarate_seq) # calculate throughput for an episode
        Worstuser_TP = np.sum(WorstuserRate_seq) # calculate throughput of the worst user for an episode
        Through_put_seq[episode] = Through_put # save throughput for an episode
        Worstuser_TP_seq[episode] = Worstuser_TP # save throughput of the worst user for an episode
        print('Episode =',episode,'Epsilon =',Epsilon,'Punishment =',punishment_counter,'Through_put =',Through_put)
    #保存学习后的模型
    agent.save(model_path.joinpath("NOMA_agent"))
    # save data
    np.save(data_path.joinpath("Through_put_NOMA.npy"), Through_put_seq)
    np.save(data_path.joinpath("WorstUser_Through_put_NOMA.npy"), Worstuser_TP_seq)
    np.save(data_path.joinpath("Total Data Rate_NOMA.npy"), datarate_seq)
    np.save(data_path.joinpath("PositionOfUsers_end_NOMA.npy"),env.PositionOfUsers)
    np.save(data_path.joinpath("PositionOfUAVs_end_NOMA.npy"), env.PositionOfUAVs)

    # print throughput
    x_axis = range(1,Episodes_number+1)
    plt.plot(x_axis, Through_put_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Throughput')
    plt.savefig(figure_path.joinpath("Throughput_NOMA.png"))
    plt.show()

    # print throughput of worst users
    x_axis = range(1,Episodes_number+1)
    plt.plot(x_axis, Worstuser_TP_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Throughput of Worst User')
    plt.savefig(figure_path.joinpath("WorstUser_Through_put_NOMA.png"))
    plt.show()

    # print datarate of the last episode(test episode when Epsilon = 0)
    x_axis_T = range(1, T+1)
    plt.plot(x_axis_T, datarate_seq)
    plt.xlabel('Steps in test epsodes')
    plt.ylabel('Data Rate of System')
    plt.savefig(figure_path.joinpath("Total_Data_Rate_NOMA.png"))
    plt.show()

if __name__ == '__main__':
    main()