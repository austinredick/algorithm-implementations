import random as rand
import numpy as np

class QLearner(object):
    def __init__(self,num_states=100,num_actions=4,alpha=0.2,gamma=0.9,rar=0.5,radr=0.99,dyna=0,verbose=False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.s = 0
        self.a = 0
        self.Q = np.zeros((self.num_states,self.num_actions))

        self.R = np.ones((self.num_states,self.num_actions)) * -1
        self.T_c = np.zeros((self.num_states, self.num_actions, self.num_states))

        self.observed_sa = set()

    def query(self, s_prime, r):
        """
        Update the Q table and return an action
        """
        s = self.s
        a = self.a

        # update Q_table
        current_q = self.Q[s,a]
        future_r = np.max(self.Q[s_prime,:])
        self.Q[s, a] = (1 - self.alpha) * current_q + self.alpha * (r + self.gamma * future_r)

        self.observed_sa.add((self.s, self.a))

        # dyna
        if self.dyna > 0:
            #update T and R
            self.T_c[s,a, s_prime] += 1
            self.R[s,a] = (1- self.alpha) * self.R[s,a] + self.alpha * r

            self.hallucinate()

        # determine action & update s,a
        if np.random.rand() < self.rar:
            self.a = rand.randint(0, self.num_actions - 1)
        else:
            self.a = np.argmax(self.Q[s_prime, :])
        self.s = s_prime

        self.rar *= self.radr

        if self.verbose:
            print(f's= {s}, a = {a}, s_prime={s_prime}, r={r}, a_prime={self.a}')
        return self.a

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table
        """
        self.s = s
        self.a = rand.randint(0, self.num_actions - 1)
        if self.verbose:
            print(f'Setting state: s = {s}, a = {self.a}')
        return self.a

    def hallucinate(self):
        """
        hallucinate additional training examples
        """
        for _ in range(self.dyna):
            s, a = rand.choice(list(self.observed_sa))
            if self.verbose:
                print(f's,a = {s,a,}... based on {self.observed_sa}')
                print(f'T_c: {self.T_c[s,a,:]}')
            T = self.T_c[s,a,:] / np.sum(self.T_c[s,a,:])
            s_prime = np.random.choice(range(self.num_states), p=T)
            r = self.R[s,a]

            current_q = self.Q[s,a]
            future_r = np.max(self.Q[s_prime,:])
            self.Q[s, a] = (1 - self.alpha) * current_q + self.alpha * (r + self.gamma * future_r)


