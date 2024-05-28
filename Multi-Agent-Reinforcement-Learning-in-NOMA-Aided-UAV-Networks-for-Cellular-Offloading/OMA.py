import numpy as np
import math
import pandas as pd
import warnings
import copy
import matplotlib.pyplot as plt
from DQN import DQN
import user_association as ua
import pathlib


# ================================ #
# 定义一些常数
#random seed control if necessary
#np.random.seed(1)

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
NumberOfUsers = NumberOfUAVs*UserNumberPerCell
F_c = 2 # carrier frequency/GHz
Bandwidth = 15 #khz
R_require = 0.1 # QoS data rate requirement kb
Power_level= 3 # Since DQN can only solve discrete action spaces, we set several discrete power gears, Please note that the change of power leveal will require a reset on the action space

amplification_constant = 10000 # Since the original power and noise values are sometims negligible, it may cause NAN data. We perform unified amplification for both signal and noise to avoid data type errors
UAV_power_unit = 100 * amplification_constant # 100mW=20dBm
NoisePower = 10**(-9) * amplification_constant # noise power
# ================================ #


class SystemModel(object):
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
        self.PositionOfUAVs.iloc[2, :] = [100, 100,100]  # UAVs' initial z
        # Initialize users and users' location
        self.User_number = NumberOfUsers
        self.K_idx = np.arange(NumberOfUsers) # set up serial number for users
        self.PositionOfUsers = pd.DataFrame(
            np.random.random((3,NumberOfUsers)),
            columns=self.K_idx.tolist(),    # Data frame for saving users' position
        )
        self.PositionOfUsers.iloc[0,:] = [204.91, 493.51, 379.41, 493.46, 258.97, 53.33]
        self.PositionOfUsers.iloc[1, :] = [219.75, 220.10, 49.81, 118.10, 332.59, 183.11]
        self.PositionOfUsers.iloc[2, :] = 0 # users' hight is assumed to be 0
        # record initial state
        self.Init_PositionOfUsers = copy.deepcopy(self.PositionOfUsers)
        self.Init_PositionOfUAVs = copy.deepcopy(self.PositionOfUAVs)
        # initialize a array to store state
        self.State = np.zeros([1, NumberOfUAVs * 3 + NumberOfUsers], dtype=float)
        # Create a data frame for storing transmit power
        self.Power_allocation_list = pd.DataFrame(
            np.ones((1, NumberOfUsers)),  # chushihua
            columns=np.arange(NumberOfUsers).tolist(),  # actions's name
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
        # Create a data frame to save SINR
        self.SINR_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)
        # Create a data frame to save datarate
        self.Daterate = pd.DataFrame(
        np.zeros((1, self.User_number)),
        columns=np.arange(self.User_number).tolist(),)
        # amplification_constant as mentioned above
        self.amplification_constant = amplification_constant


    def User_randomMove(self,MAXspeed,NumberofUsers):
        """
        使用随机算法移动用户
        users random move
        Args:
            MAXspeed (float): 最大移动速率
            NumberofUsers (int): 用户数量
        """
        self.PositionOfUsers.iloc[[0,1],:] += np.random.randn(2,NumberofUsers)*MAXspeed
        return


    def Get_Distance_U2K(self,UAV_Position,User_Position,UAVsnumber,Usersnumber): 
        """
        计算user和UAV之间的距离
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
                # calculate Distence betwen UAV i and User j
                self.Distance.iloc[i,j] = np.linalg.norm(UAV_Position.iloc[:,i]-User_Position.iloc[:,j]) 
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
        for i in range(UAVsnumber):
            for j in range(Usersnumber):
                UAV_Hight=UAV_Position.iloc[2,i]
                # calculate distance
                D_H = np.sqrt(np.square(distance_U2K.iloc[i,j])-np.square(UAV_Hight)) 
                # calculate the possibility of LOS/NLOS
                d_0 = np.max([(294.05*math.log(UAV_Hight,10)-432.94),18])
                p_1 = 233.98*math.log(UAV_Hight,10) - 0.95
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
                # average pathloss
                Avg_Los = P_Los*L_Los + P_NLos*L_NLos 
                gain = np.random.rayleigh(scale=1, size=None)*pow(10,(-Avg_Los/10)) # random fading
                self.Propergation_Loss.iloc[i,j] = gain #save pathloss
        return self.Propergation_Loss


    def Get_SINR_OMA(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power):
        """
        计算每个用户的SINR值
        This function is to calculate the SINR for every users
        Args:
            UAVsnumber (int): UAV的数量
            Usersnumber (int): 用户数量
            PropergationLosslist (array): 传播损失数组
            UserAssociationlist (array): 用户关联数组
            Noise_Power (float): 噪声功率
        Returns:
            array: SINR数组
        """
        for j in range(Usersnumber): # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service 'j_idx' represents the other users
            i_Server_UAV = UserAssociationlist.iloc[0,j]
            Signal_power = self.Power_allocation_list.iloc[0,j] * PropergationLosslist.iloc[i_Server_UAV,j] # read the sinal power from power allocation list
            I_inter_cluster = 0
            for j_idx in range(Usersnumber): # calculate Interference for user j
                if UserAssociationlist.iloc[0,j_idx] == i_Server_UAV:
                    pass
                else:
                    Inter_UAV = UserAssociationlist.iloc[0,j_idx] # calculate inter cluster interference from other UAVs
                    I_inter_cluster = I_inter_cluster + (self.Power_allocation_list.iloc[0,j_idx] * PropergationLosslist.iloc[Inter_UAV,j])/ (Usersnumber/UAVsnumber)#
            SINR = Signal_power/(I_inter_cluster + Noise_Power) # calculate SINR and save it
            self.SINR_list.iloc[0,j] = SINR

        return self.SINR_list


    def Get_Channel_Gain_OMA(self,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power):
        """
        计算信道增益
        this function is for calculating channel gain
        Args:
            Usersnumber (int): 用户数量
            PropergationLosslist (array): 传播损失数组
            UserAssociationlist (array): 用户关联数组
            Noise_Power (float): 噪声功率
        Returns:
            array: 信道增益数组
        """
        for j in range(Usersnumber):   # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = 10000 * PropergationLosslist.iloc[i_Server_UAV, j]
            I_inter_cluster = 0
            ChannelGain = Signal_power / (I_inter_cluster + Noise_Power)# calculate channel gain
            self.ChannelGain_list.iloc[0, j] = ChannelGain # save channel gain
        return self.ChannelGain_list


    def Calculate_Datarate(self,SINRlist,Usersnumber,B):
        """
        计算所有用户的速率
        calculate data rate for all users
        Args:
            SINRlist (array): SINR数组
            Usersnumber (int): 用户数量
            B (int): 带宽
        Returns:
            array: 用户速率数组
        """
        for j in range(Usersnumber):
            if SINRlist.iloc[0,j] <=0:
                print(SINRlist)
                warnings.warn('SINR wrong') # A data type error may occur when the data rate is too small, thus we ste up this alarm
            self.Daterate.iloc[0,j] = B*math.log((1+SINRlist.iloc[0,j]),2)
        SumDataRate = sum(self.Daterate.iloc[0,:])
        Worst_user_rate = min(self.Daterate.iloc[0,:])
        return self.Daterate,SumDataRate,Worst_user_rate


    def Reset_position(self):
        """
        重置用户和UAV位置
        save initial state for environment reset
        """
        self.PositionOfUsers = copy.deepcopy(self.Init_PositionOfUsers)
        self.PositionOfUAVs = copy.deepcopy(self.Init_PositionOfUAVs)
        return


    def Create_state_Noposition(self,serving_UAV,User_association_list,User_Channel_Gain):
        """
        创建状态
        Create state, pay attention we need to ensure UAVs and users who are making decisions always input at the fixed neural node to achieve MDQN
        状态包含 UAV位置和CSI信息
        Args:
            serving_UAV (int): 选择的UAV
            User_association_list (array): 用户关联数组
            User_Channel_Gain (array): 信道增益数组
        Returns:
            array: 状态
        """
        UAV_position_copy = copy.deepcopy(self.PositionOfUAVs.values)
        UAV_position_copy[:,[0,serving_UAV]] = UAV_position_copy[:,[serving_UAV,0]] # adjust the input node of serving UAV to ensure it is fixed
        User_Channel_Gain_copy = User_Channel_Gain.values[0]
        for UAV in range(NumberOfUAVs):
            self.State[0, 3 * UAV:3 * UAV + 3] = UAV_position_copy[:, UAV].T # save UAV positions as a part of the state
        User_association_copy = copy.deepcopy(User_association_list.values)
        desirable_user = np.where(User_association_copy[0]==serving_UAV)[0] # find out the current served users
        for i in range(len(desirable_user)):
            User_Channel_Gain_copy[i],User_Channel_Gain_copy[desirable_user[i]] = User_Channel_Gain_copy[desirable_user[i]],User_Channel_Gain_copy[i] # Similarly, adjust the input node of the current served users
        for User in range(NumberOfUsers):
            self.State[0,(3*UAV+3)+User] = User_Channel_Gain_copy[User].T # save CSI of users in state
        Stat_for_return = copy.deepcopy(self.State)
        return Stat_for_return


    def take_action(self,action_number,acting_UAV,User_asso_list):
        """
        Args:
            action_number (int): 其实就是action index
            acting_UAV (int): 执行的UAV index
            User_asso_list (array): 用户关联数组
        """
        UAV_move_direction = action_number % 7 #UAV has seven positional actions
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
        # Power allocation part - OMA
        power_allocation_scheme = action_number//7
        acting_user_list = np.where(User_asso_list.iloc[0,:] == acting_UAV)[0]
        # 同一个UAV连接的两个用户
        # three power levels for each user
        User0 = acting_user_list[0]
        User1 = acting_user_list[1]
        
        # for users, the power levels can be 1/4, 1, 4 * power unit
        if power_allocation_scheme % 3 == 0:
            self.Power_allocation_list.iloc[0,User0] = self.Power_unit
        elif power_allocation_scheme % 3 == 1:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit/4
        elif power_allocation_scheme % 3 == 2:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*4

        if power_allocation_scheme // 3 == 0:
            self.Power_allocation_list.iloc[0,User1] = self.Power_unit
        elif power_allocation_scheme // 3 == 1:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/4
        elif power_allocation_scheme // 3 == 2:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit*4


def main():
    Episodes_number = 150 # total episodes number
    Test_episodes_number = 30  # number of test episodes
    T = 60 #total time slots (steps)
    T_AS = np.arange(0, T, 200) # time slot of user association, current setting indicate 1
    print("time slots",T_AS)
    env = SystemModel()
    agent = DQN(UserNumberPerCell,NumberOfUAVs,NumberOfUsers) # create an agent
    Epsilon = 0.9
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate
    WorstuserRate_seq = np.zeros(T) # Initialize memory to store data rate of the worst user
    Through_put_seq = np.zeros(Episodes_number) # Initialize memory to store throughput
    Worstuser_TP_seq = np.zeros(Episodes_number) # Initialize memory to store throughput of the worst user
    path = pathlib.Path(__file__).parent.resolve()
    agent.save(path.joinpath("model").joinpath("OMA_init"))
    for episode in range(Episodes_number):
        env.Reset_position()
        #if Epsilon > 0.05: # determine the minimum Epsilon value
        Epsilon -= 0.9 / (Episodes_number - Test_episodes_number) # decaying epsilon
        p = 0 # punishment counter
        for t in range(T):
            # 原来的数组是[0]，结合代码就相当于每个episode刚开始的时候做一次
            if t in T_AS:
                User_AS_List = ua.kmeans(env.PositionOfUAVs, env.PositionOfUsers,NumberOfUAVs, NumberOfUsers) # user association after each period because users are moving
            for UAV in range(NumberOfUAVs):
                # 1.计算距离，计算信道衰落，计算信道增益，计算状态，Agent预测动作，执行动作
                Distance_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, NumberOfUsers) # Calculate the distance for each UAV-users
                PL_for_CG = env.Get_Propergation_Loss(Distance_CG, env.PositionOfUAVs, NumberOfUAVs, NumberOfUsers, F_c) # Calculate the pathloss for each UAV-users
                CG = env.Get_Channel_Gain_OMA(NumberOfUsers, PL_for_CG, User_AS_List,NoisePower)  # Calculate the channel gain for each UAV-users
                State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
                action_name = agent.Choose_action(State,Epsilon) # agent calculate action
                env.take_action(action_name,UAV,User_AS_List) # take action in the environment
                # 2.计算执行动作之后的情况
                Distance = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, NumberOfUsers) # after taking actions, calculate the distance again
                P_L = env.Get_Propergation_Loss(Distance,env.PositionOfUAVs,NumberOfUAVs, NumberOfUsers, F_c) #calculate the pathloss
                SINR=env.Get_SINR_OMA(NumberOfUAVs,NumberOfUsers,P_L,User_AS_List,NoisePower) # calculate SINR for users
                DataRate, SumRate,WorstuserRate = env.Calculate_Datarate(SINR, NumberOfUsers, Bandwidth) # calculate data rate, sum rate and the worstusers data rate
                # calculate raward based on sum rate and check if users meet the QOS requirement
                # 3.计算Reward
                Reward = SumRate
                if WorstuserRate < R_require:
                    Reward = Reward/2
                    p += 1
                # 4.下一个状态情况
                CG_next = env.Get_Channel_Gain_OMA(NumberOfUsers, P_L, User_AS_List,NoisePower) # Calculate the equivalent channel gain for S_{t+1}
                Next_state = env.Create_state_Noposition(UAV,User_AS_List,CG_next) # Generate S_{t+1}
                # 5.组织经验
                #copy data for (S_t,A_t,S_t+1,R_t)
                State_for_memory = copy.deepcopy(State[0])
                Action_for_memory = copy.deepcopy(action_name)
                Next_state_for_memory = copy.deepcopy(Next_state[0])
                Reward_for_memory = copy.deepcopy(Reward)
                # 存储并训练
                agent.remember(State_for_memory, Action_for_memory, Next_state_for_memory, Reward_for_memory) #save the MDP transitions as (S_t,A_t,S_t+1,R_t)
                agent.train() #train the DQN agent
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
        print('Episode=',episode,'Epsilon=',Epsilon,'Punishment=',p,'Through_put=',Through_put)
    agent.save(path.joinpath("model").joinpath("OMA_agent"))
    # save data
    np.save("data/Through_put_OMA.npy", Through_put_seq)
    np.save("data/WorstUser_Through_put_OMA.npy", Worstuser_TP_seq)
    np.save("data/Total Data Rate_OMA.npy", datarate_seq)
    np.save("data/PositionOfUsers_end_OMA.npy",env.PositionOfUsers)
    np.save("data/PositionOfUAVs_end_OMA.npy", env.PositionOfUAVs)

    # print throughput
    x_axis = range(1,Episodes_number+1)
    plt.plot(x_axis, Through_put_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Though put')
    plt.savefig('figure/Through_put_OMA.png')
    plt.show()

    # print throughput of worst users
    x_axis = range(1, Episodes_number+1)
    plt.plot(x_axis, Worstuser_TP_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Though put of Worst User')
    plt.savefig('figure/WorstUser_Through_put_OMA.png')
    plt.show()

    # print datarate of the last episode(test episode when Epsilon = 0)
    x_axis_T = range(1, T+1)
    plt.plot(x_axis_T, datarate_seq)
    plt.xlabel('Steps in test epsodes')
    plt.ylabel('Data Rate of System')
    plt.savefig('figure/Total Data Rate_OMA.png')
    plt.show()

if __name__ == '__main__':
    main()