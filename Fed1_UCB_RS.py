# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def fp(p):
    fp = 10*np.log(T)
    return int(fp)

def gp(p):
    if p == 1:
        gp = 2000
    else:
        gp = 0          
    return int(gp)


N = 100 #repeat times
global T 
T = int(1e6)
sigma = 1/5
sigma_c = 1/10

comm_c = 0
regret = np.zeros([N,T])

movie = np.load('movielens_norm_100.npy')

K = movie.shape[1]

for rep in range(N):
    t = 0
    p = 1
    M = 0
    
    active_arm = np.array(range(K),dtype = int)
    C = 1 #communication loss
    reward_global = np.zeros(T)
    optimal_reward = np.zeros(T)
    
    
    mu_global = sum(movie)/movie.shape[0]
    optimal_index = np.where(mu_global==np.max(mu_global))
 
    while t<T:
        '''
        round p
        '''
        
        '''
        local players
        '''
        if len(active_arm)>1:
            player_add_num = gp(p)
            if M==0:
                M += 1
                pull_num = np.zeros([1,K])
                reward_local = np.zeros([1,K])
                mu_local = np.zeros([1,K])
                for k in range(K):
                    mu_local[M-1,k] = np.random.normal(mu_global[k], sigma_c) #generated local mean
                player_add_num -= 1
            
            for m in range(player_add_num):
                M += 1
                pull_num = np.r_[pull_num, np.zeros([1,K])]
                reward_local = np.r_[reward_local, np.zeros([1,K])]
                mu_local = np.r_[mu_local, np.zeros([1,K])]
                for k in range(K):
                    mu_local[M-1,k] = np.random.normal(mu_global[k], sigma_c) #generated local mean
        
        expl_len = fp(p)
        p += 1
        
        if len(active_arm)>1:
            for k in active_arm:
                for _ in range(min(T-t,expl_len)):
                    for m in range(M):
                        reward_local[m,k] += np.random.normal(mu_local[m,k],sigma)
                        pull_num[m,k] += 1
                    reward_global[t] = reward_global[t-1]+M*np.random.normal(mu_global[k],sigma_c)
                    optimal_reward[t] = optimal_reward[t-1]+M*np.random.normal(mu_global[optimal_index][0],sigma_c)
                    t = t+1
            mu_local_sample = reward_local/pull_num
        
        if len(active_arm)==1:
            reward_global[t:] = reward_global[t-1]+np.arange(T-t)*M*mu_global[active_arm[0]]
            optimal_reward[t:] = optimal_reward[t-1]+np.arange(T-t)*M*mu_global[optimal_index]
            break
        '''
        global server
        '''
        if len(active_arm)>1:
            reward_global[t-1] -= M*C #comment this line out to ignore communication loss
            comm_c += M
            E = np.array([])
            mu_global_sample = 1/M*sum(mu_local_sample)
            eta_p = 0
            for i in range(1,p): # p has been added one above
                F_d = 0
                for j in range(i,p):
                    F_d += fp(j)
                eta_p += 1/M**2*gp(i)/F_d
            
            conf_bnd = np.sqrt(sigma**2*eta_p*np.log(T)) #the constants are tuned from the original ones in the paper to get better performance
            elm_max = np.nanmax(mu_global_sample)-conf_bnd
            for index in range(len(active_arm)):
                arm = active_arm[index]
                if mu_global_sample[arm]+conf_bnd<elm_max:
                    E = np.append(E,np.array([arm]))
        
            for i in range(len(E)):
                active_arm = np.delete(active_arm, np.where(active_arm == E[i]))
            
    regret[rep] = optimal_reward-reward_global

plt.figure()
avg_regret = 1/N*sum(regret)
err_regret = np.sqrt(np.var(abs(avg_regret-regret),axis=0))
print('regret:',avg_regret[-1],'comm:', comm_c/N, 'delta:',round(np.sort(mu_global)[-1]-np.sort(mu_global)[-2],3))
plt.plot(range(T),avg_regret)