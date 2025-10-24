# This file runs MCMC on the quadratic model using Stan, and saves output to a file as reference samples
import stan
import matplotlib.pyplot as plt
import numpy as np

model = """
data {
  real y;  // observation value 
}
parameters {
  real theta1;       
  real theta2;      
}
model {
   theta1 ~ normal(0,1); // prior theta1
   theta2 ~ normal(0,1); // prior theta2
   y ~ normal(theta1-square(theta2), 0.01); // likelihood
 }
"""
quadratic_obs = {'y':0}

posterior = stan.build(model, data=quadratic_obs)
fit = posterior.sample(
    num_chains = 4,
    num_samples = 10_000,
    delta = 0.95
)

theta1 = fit['theta1'][0]
theta2 = fit['theta2'][0]
# print(fit.to_frame().describe().T)
# plt.figure('theta1 trace plot')
# plt.plot(theta1, '-')
# plt.figure('theta2 trace plot')
# plt.plot(theta2, '-')
# plt.figure('theta scatter plot')
# plt.plot(theta1, theta2, 'rx')
# plt.show()

ref_sample = np.array([theta1, theta2]).T
ref_sample = ref_sample[0:40000:20, :]

np.savetxt('quadratic_ref_sample.txt', ref_sample)