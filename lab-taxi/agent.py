import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.005
        self.gamma = 0.99
        self.alpha = 0.1
        
        # Result
        # Episode 20000/20000 || Best average reward 9.333
        # self.epsilon = 0.005
        # self.gamma = 0.99
        # self.alpha = 0.1
        # Expected-SARSA
        

    def select_epsilon_greedy_action(self, state):
        action = np.random.choice(np.arange(self.nA), p=self.get_probs(self.Q[state], self.epsilon, self.nA)) \
                      if state in self.Q else np.random.choice(np.arange(self.nA))
        return action

    def get_probs(self, Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.select_epsilon_greedy_action(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # [Q-Learning]
        # next_action = np.argmax(self.Q[next_state])
        # self.Q[state][action] += self.alpha*(reward + self.gamma*(0 if done else self.Q[next_state][next_action]) - self.Q[state][action])
        
        # [Expected-SARSA]
        best_action = np.argmax(self.Q[next_state])
        self.Q[state][action] += self.alpha*(reward + self.gamma*(0 if done else (self.Q[next_state][best_action]*(1.0 - self.epsilon) + self.epsilon/self.nA*np.sum(self.Q[next_state]))) - self.Q[state][action])