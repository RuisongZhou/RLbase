# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 2/24/2019

#用saram算法训练寻址小游戏
import pandas as pd
import numpy as np
from maze_env import Maze


class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
         super(SarsaTable,self).__init__(actions, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        predict = self.q_table.loc[s,a]

        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r

        self.q_table.loc[s,a] += self.lr * (q_target - predict)

class SarsaLambda(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,trace_decay=0.9):
        super(SarsaLambda,self).__init__(actions, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            newState = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table.append(newState)

            #update eligibility table
            self.eligibility_trace = self.eligibility_trace.append(newState)
        
    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        predict = self.q_table.loc[s,a]

        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r

        error = q_target - predict

        self.eligibility_trace.loc[s,:] *=0
        self.eligibility_trace.loc[s,a] = 1
        #update Q
        self.q_table += self.lr * error * self.eligibility_trace
        #update E
        self.eligibility_trace *=self.gamma*self.lambda_

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0
        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    env.reset()
    RL = SarsaLambda(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()