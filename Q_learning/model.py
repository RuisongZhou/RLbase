import numpy as np
import pandas as pd

from maze_env import Maze

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        #select an action 
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation,:]

            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)

        return action

    def train(self,state,action,reward,s_):
        self.check_state_exist(s_)
        predict = self.q_table.loc[state,action]

        if s_ != 'terminal':
            target = reward + self.gamma*self.q_table.loc[s_,:].max()
        else:
            target = reward
        
        self.q_table.loc[state,action] += self.lr*(target-predict)

    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append a new state
            self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


env = Maze()
RL = QLearningTable(actions=list(range(env.n_actions)))

env.after(100, update)
env.mainloop()