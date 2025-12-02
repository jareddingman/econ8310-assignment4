#imports
import pandas as pd
import pymc as pm
import scipy.stats as stats
import numpy as np
from scipy.special import expit #need this library to make trace into a np array


#load da data
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")

example_obs = data.loc[:, ['version', 'retention_1']]
example_obs['40col'] = (example_obs['version']=="gate_40").astype(int)

with pm.Model() as model:
    #retention is a yes/no, so should not use a continuous DV
    #soooo we should have a logit model where we try to evaluate the beta coefficent like DV = alpha + beta*x
    #remember this is for log odds too
    alpha = pm.Normal("alpha", mu = 0, sigma = 1)
    beta = pm.Normal("beta", mu = 0, sigma = 1)

    logit = alpha + beta*example_obs['40col'].values

    #need p for the bernoulii pm arg

    p = pm.Deterministic("p", pm.math.invlogit(logit))

    retention = pm.Bernoulli("retention", p = p, observed = example_obs['retention_1'].values)

#Training Model-----------------------------------------------
# with model:
#     step = pm.Metropolis()
#     trace = pm.sample(1000, tune=5000, step=step, return_inferencedata=True) #I got a timeout warning from Google witht the 10000
#-----------------------------------------------------------------

p30 = expit(trace.posterior['alpha'].values)
p40 = expit(trace.posterior['alpha'].values + trace.posterior['beta'].values) #need to add the alpha values to follow the above model

change = p40 - p30
#print(type(change)) 
avgChange = np.mean(change)
confidence = np.percentile(change, [2.5, 97.5])

print("The average change in retention is", avgChange)
print("The bayesian confidence interval is", confidence)





