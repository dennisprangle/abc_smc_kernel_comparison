import numpy as np
import stan
import matplotlib.pyplot as plt

model = """
data {
  vector[2] y[4];
}
parameters {
  real theta1;       
  real theta2;      
  real theta3;
  real theta4;
  real theta5;
}
model {
	vector[2] mu;
	matrix[2,2] cov;
  real s1;
  real s2;
  real rho;

	//prior
    theta1 ~ uniform(-3,3);
    theta2 ~ uniform(-3,3);
    theta3 ~ uniform(0,3);
    theta4 ~ uniform(0,3);
    theta5 ~ uniform(-3,3);
    //model
    s1 = square(theta3) + 1e-6;
    s2 = square(theta4) + 1e-6;
    rho = tanh(theta5);
    mu[1] = theta1;
    mu[2] = theta2;
    cov[1,1] = square(s1);
    cov[1,2] = rho*s1*s2;
    cov[2,1] = rho*s1*s2;
    cov[2,2] = square(s2);
    for (n in 1:4) {
    	y[n] ~ multi_normal(mu,cov);
    }
 }
"""

# The following observations were generated under the parameters (0.7, -2.9, -1, -0.9, 0.6),
# as used in the original paper
obs = {'y':np.array([
  [1.3614, -2.4299],
  [0.7617, -2.4486],
  [0.2481, -3.2101],
  [-0.8228, -3.3016]]
)}

posterior = stan.build(
	model,
	data=obs
)
fit = posterior.sample(
	num_chains = 4,
	num_samples = 10000
)

df = fit.to_frame()
print(df.describe().T)
df = df.filter(like='theta')
# for parname in df.keys():
#   plt.figure()
#   plt.plot(df[parname], '-')
#   plt.ylabel(parname)
# plt.show()

# Convert to numpy
ref_sample = df.to_numpy()[0:40000:20, :]
# Randomly flip signs of theta3 and theta4
for i in (2,3):
  ref_sample[:,i] *= np.random.choice([-1,1], size=2000)

np.savetxt('slcp_ref_sample.txt', ref_sample)
