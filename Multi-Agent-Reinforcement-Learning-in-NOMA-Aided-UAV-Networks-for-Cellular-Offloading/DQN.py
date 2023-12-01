import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
import copy
import matplotlib.pyplot as plt


class DQN(object):
    def __init__(self,UserNumberPerCell,NumberOfUAVs,NumberOfUsers):
        self.update_freq = 600  # Model update frequency of the target network
        self.replay_size = 10000  # replay buffer size
        self.step = 0
        self.replay_queue = deque(maxlen=self.replay_size)

        self.power_number = 3 ** UserNumberPerCell # 3 power actions
        self.action_number = 7 * self.power_number # 7 positional actions

        self.NumberOfUAVs = NumberOfUAVs
        self.NumberOfUsers = NumberOfUsers
        
        self.model = self.create_model() # crate model
        self.target_model = self.create_model() # crate target model

    def create_model(self):
        #Create a neural network with a input, hidden, and output layer
        STATE_DIM = self.NumberOfUAVs*3 + self.NumberOfUsers # input layer dim
        ACTION_DIM = 7 * self.power_number # output layer dim
        model = models.Sequential([
            layers.Dense(40, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(learning_rate=0.001)) #Set the optimmizer and learning rate here
        return model


    def Choose_action(self, s, epsilon):
        # Choose actions according to e-greedy algorithm
        if np.random.uniform() < epsilon:
            return np.random.choice(self.action_number)
        else:
            return np.argmax(self.model.predict(s))


    def remember(self, s, a, next_s, reward):
        # save MDP transitions
        self.replay_queue.append((s, a, next_s, reward))


    def train(self,batch_size = 128, lr=1 ,factor = 1):
        if len(self.replay_queue) < self.replay_size:
            return # disable learning until buffer full
        self.step += 1

        # Over 'update_freq' steps, assign the weight of the model to the target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch) # calculate Q value
        Q_next = self.target_model.predict(next_s_batch) # predict Q value

        # update Q value following bellamen function
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        self.model.fit(s_batch, Q, verbose=0)   # DNN training
    
    def save(self,name):
        self.model.save(name)