#%% 
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats
az.style.use("arviz-darkgrid")

#to be re-checked
def pminT():
    #compute the pminT update, assuming the bottom time series
    #to have equal mean and variance
    prior_x = [lambda_b , lambda_b]
    var= lambda_b
    sigma_u = np.sqrt(lambda_u)
    inc = lambda_u - sum(prior_x)  
    g = np.array([var,  var]) * 1/(lambda_u + var + var)
    post_x = prior_x + g  *  inc
    post_var = var * (1 - g) 
    prob_neg = norm.cdf(0, loc = post_x[0], scale = np.sqrt(post_var[0]))
    return (post_x[0], prob_neg)

#sample numerically the posterior poisson
def poisson_prog():
    basic_model = pm.Model()
    with basic_model:
        x1 = pm.Poisson ('x1', mu = lambda_b)
        x2 = pm.Poisson ('x2', mu = lambda_b)
        u =  pm.Poisson ('u',  mu = lambda_u, observed = x1 + x2)
        
    with basic_model:
        trace = pm.sample(5000)
    
    #the two posterior mean are numerically slightly different
    #we average them            
    pois_mean = np.mean(az.summary(trace)["mean"])
    return (pois_mean)

#to check how much it takes to reconcile a monthly hierarchy
#it takes about 30 sec on my Mac
def poisson_prog_monthly():
    basic_model = pm.Model()
    with basic_model:
        x1 = pm.Poisson ('x1', mu = lambda_b)
        x2 = pm.Poisson ('x2', mu = lambda_b)
        x3 = pm.Poisson ('x3', mu = lambda_b)
        x4 = pm.Poisson ('x4', mu = lambda_b)
        x5 = pm.Poisson ('x5', mu = lambda_b)
        x6 = pm.Poisson ('x6', mu = lambda_b)
        x7 = pm.Poisson ('x7', mu = lambda_b)
        x8 = pm.Poisson ('x8', mu = lambda_b)
        x9 = pm.Poisson ('x9', mu = lambda_b)
        x10 = pm.Poisson ('x10', mu = lambda_b)
        x11 = pm.Poisson ('x11', mu = lambda_b)
        x12 = pm.Poisson ('x12', mu = lambda_b)
        q1 =  pm.Poisson ('q1',  mu = 3 * lambda_b, observed = x1 + x2 + x3 + x4)
        q2 =  pm.Poisson ('q2',  mu = 3 * lambda_b, observed = x5 + x6 + x7 + x8)
        q3 =  pm.Poisson ('q3',  mu = 3 * lambda_b, observed = x9 + x10 + x11 + x12)
        s1 =  pm.Poisson ('s1',  mu = 5 * lambda_b, observed = x1 + x2 + x3 + x4 + x5 + x6)
        s2 =  pm.Poisson ('s2',  mu = 8 * lambda_b, observed = x7 + x8 + x9 + x10 + x11 + x12)
        y =  pm.Poisson  ('y',   mu = 14 *lambda_b,  observed = x1 + x2 + x3 + x4 + x5 + x6 +x7 + x8 + x9 + x10 + x11 + x12)
        

        
    with basic_model:
        trace = pm.sample(5000)
    
    #the two posterior mean are numerically slightly different
    #we average them            
    pois_mean = np.mean(az.summary(trace)["mean"])
    return (pois_mean)


#%% 
#sample numerically the posterior skew normal.
#we now estimate by MLE on sampled data, but it can be done by moment matching 
def skew_normal_prog():
    sample_b = poisson.rvs ( lambda_b, size=2500 )
    sample_u = poisson.rvs ( mu= lambda_u, size=2500 )

    a_b, loc_b, scale_b = stats.skewnorm.fit(sample_b)
    a_u, loc_u, scale_u = stats.skewnorm.fit(sample_u)
    
    basic_model = pm.Model()
    with basic_model:
        x1 = pm.SkewNormal ( 'x1', mu = loc_b, sigma = scale_b, alpha = a_b )
        x2 = pm.SkewNormal ( 'x2', mu = loc_b, sigma = scale_b, alpha = a_b )
        u =  pm.SkewNormal ( 'u',  mu = loc_u, sigma= scale_u,  alpha = a_u, observed = x1 + x2)
        
    with basic_model:
        trace = pm.sample(5000)
    
    
    #the two posterior mean are numerically slightly different
    #we average them            
    skew_mean = np.mean(az.summary(trace)["mean"])
    neg_x1 = np.mean(trace.get_values('x1')<0)
    neg_x2=  np.mean(trace.get_values('x2')<0)
    skew_prob = np.mean([neg_x1, neg_x2])
    return (skew_mean, skew_prob)



#%% 

# lambdas = [1, 1.5, 2.5]
lambdas = [1]

#incoherence = [+2, -2]
incoherence = [-2]
counter = 0
df = pd.DataFrame(index= range(len(lambdas) * len (incoherence)), 
    columns=['lambda_b','lambda_u','incoherence','Ps_mean','mean_mint', 'p_neg mint',
             'mean_skew', 'p_neg_skew'])
#%% 
    
for lambda_b in lambdas:
    print("lambda_b: " + str(lambda_b))
    for current_inc in incoherence:
        lambda_u = np.max([(2 * lambda_b) + current_inc, 0.5]) 
        print("lambda_u: " + str(lambda_u))
   
        
        
        #nested function access lambda_b and lambda_u automatically
        mint_mean, mint_prob = pminT()
        pois_mean = poisson_prog()
        skew_mean, skew_prob = skew_normal_prog()
        

        df['lambda_b'][counter] = lambda_b
        df['lambda_u'][counter] = lambda_u
        df['incoherence'][counter] = lambda_u - 2 * lambda_b
        df['Ps_mean'][counter] = pois_mean
        df['mean_mint'][counter] = mint_mean
        df['p_neg mint'][counter] = mint_prob
        df['mean_skew'][counter] = skew_mean
        df['p_neg_skew'][counter] = skew_prob
        counter = counter + 1


