
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:54:07 2023

@author: Cecilia
"""

import torch
from torch.distributions import binomial, poisson

class SIRModel():
    def __init__(self, max_time=100, time_step=5, N=1000, I0=1):
        #max_time: length of the observe sequence
        #N: size of the population
        #I0: nr of infectious people at the beginning
        self.max_time = max_time
        self.N = N
        self.I0 = I0
        self.time_step = time_step

    def simulate(self, inputs):
        #Observed data are the number of infectious people at each time step
        #Inputs are:
        #           gamma, recovery rate
        #           beta, infection rate
        S = torch.zeros(inputs.shape[0],self.max_time)
        I = torch.zeros(inputs.shape[0],self.max_time)
#        R = torch.zeros(inputs.shape[0],self.max_time)
        S[:,0] = self.N-self.I0
        I[:,0] = self.I0
        gamma = inputs[:,0]
        beta = inputs[:,1] 
        pr_rec = 1.0 - torch.exp(-gamma)                                       #to transform the rate in probability
        for j in range(1,self.max_time):
            lamb = beta*I[:,j-1]/self.N
            pr_inf = 1.0 - torch.exp(-lamb)                                    #to transform the rate in probability
            infection = binomial.Binomial(S[:,j-1],pr_inf).sample()
            recovery = binomial.Binomial(I[:,j-1],pr_rec).sample()
            S[:,j] = S[:,j-1]-infection
            I[:,j] = I[:,j-1]+infection-recovery
#            R[:,j] = R[:,j-1]+recovery
        #TODO: return also S and R? 
        return torch.index_select(I,1,torch.arange(0,self.max_time,self.time_step))
    
class SEIRModel():
    def __init__(self, max_time=100, time_step=5, N=1000, S0=999, E0=1, R0=0):
        #max_time: length of the observe sequence
        #N: size of the population
        #I0: nr of infectious people at the beginning
        self.max_time = max_time
        self.N = N
        self.S0 = S0
        self.E0 = E0
        self.R0 =R0
        self.time_step = time_step

    def simulate(self, inputs):
        #Observed data are the number of infectious people at each time step
        #Inputs are:
        #           gamma, recovery rate
        #           beta, exposure rate
        #           alpha, infection rate

        S = torch.zeros(inputs.shape[0],self.max_time)
        I = torch.zeros(inputs.shape[0],self.max_time)
        E = torch.zeros(inputs.shape[0],self.max_time)
        incidence = torch.zeros(inputs.shape[0],self.max_time) # new infection count
    #   R = torch.zeros(inputs.shape[0],self.max_time)
        S[:,0] = self.S0
        E[:,0] = self.E0
        I[:,0] = self.N-self.S0-self.E0-self.R0
        gamma = inputs[:,0]
        beta = inputs[:,1]
        alpha = inputs[:,2]
        pr_rec = 1.0 - torch.exp(-gamma) 
        pr_inf = 1.0 - torch.exp(-alpha)                                      #to transform the rate in probability
        for j in range(1,self.max_time):
            lamb = beta*I[:,j-1]/self.N
            pr_exp = 1.0 - torch.exp(-lamb)                                    #to transform the rate in probability
            exposion = binomial.Binomial(S[:,j-1],pr_exp).sample()
            incidence[:,j] = binomial.Binomial(E[:,j-1],pr_inf).sample()
            recovery = binomial.Binomial(I[:,j-1],pr_rec).sample()
            S[:,j] = S[:,j-1]-exposion
            E[:,j] = E[:,j-1]+exposion-incidence[:,j]
            I[:,j] = I[:,j-1]+incidence[:,j]-recovery
    #       R[:,j] = R[:,j-1]+recovery
        #TODO: return also S and R? 
        obs_rate = 0.5
        background = 0.1
        return torch.poisson(background + obs_rate*torch.index_select(incidence,1,torch.arange(0,self.max_time,self.time_step)))