import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists
from particles import mcmc
from matplotlib import pyplot as plt

"We use the 'particles' package to implement PMMH for a state-space model"
"See https://github.com/nchopin/particles"
" and https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/index.html"

# Definition of the seir model as a State Space Model
    
class seir(ssm.StateSpaceModel):
    # parameters are in log scale
    # gamma is the recovery rate
    # beta is the infection rate
    # alpha is the exposure rate
    # N is the population size (default 1000)
    # the default initial state of the system is: 990 susceptible and 10 exposed individuals
    default_params = {"lngamma": -3, "lnbeta": -1, "lnalpha": -0.5, "N":1000, "S0":990, "E0":10, "R0":0}
    
    # Distribution to sample initial state
    # Here, is a Dirac dist
    def PX0(self): 
        chain_rule = dists.OrderedDict()
        chain_rule['S'] = dists.Dirac(loc=self.S0)
        chain_rule['E'] = dists.Dirac(loc=self.E0)
        chain_rule['R'] = dists.Dirac(loc=self.R0)
        chain_rule['incidence'] = dists.Dirac(loc=0) # New infectious cases this time step
        return dists.StructDist(chain_rule)
    
    # Distribution of X_t given X_{t-1} 
    def PX(self, t, xp):
        alpha = np.exp(self.lnalpha)
        beta = np.exp(self.lnbeta)
        gamma = np.exp(self.lngamma)
        N = self.N
        S = xp['S'].astype(int)
        E = xp['E'].astype(int)
        R = xp['R'].astype(int)
        I = N-(S+E+R)
        pi_se = 1-np.exp(-beta*I/N)
        pi_ei = 1-np.exp(-alpha)
        pi_ir = 1-np.exp(-gamma)
        se = dists.Binomial(S, pi_se)
        ir = dists.Binomial(I, pi_ir)
        ei = dists.Binomial(E, pi_ei) # new incidences
        Snew = dists.LinearD(se, a=-1, b=S) # S -> S - se
        Enew = dists.Cond(
            lambda x: dists.Dirac(loc = E + S - x['S'] - x['incidence']) # E -> E + se - ei
        )
        Rnew = dists.LinearD(ir, a=1, b=R) # R -> R + ir
        chain_rule = dists.OrderedDict()
        chain_rule['S'] = Snew
        chain_rule['incidence'] = ei
        chain_rule['E'] = Enew
        chain_rule['R'] = Rnew
        return dists.StructDist(chain_rule)
    
    # Distribution of Y_t given X_t=x, and X_{t-1}=xp
    def PY(self, t, xp, x):
        incidence = x['incidence'].astype(int)
        obs_rate = 0.5
        background = 0.1
        return dists.Poisson(rate=background+incidence*obs_rate)

# Prior is independent Normals
prior = dists.StructDist({
    "lngamma":dists.Normal(-3., scale=2.),
    "lnbeta":dists.Normal(-1., scale=2.),
    "lnalpha": dists.Normal(-0.5, scale=2.)
})

#Observations over T days
T = 30
np.random.seed(10)
obs = seir(lngamma = -2.6, lnbeta = -0.2, lnalpha = 0.0).simulate(T)[1]
#print(np.array([ i[0] for i in obs ])) # Print a more human readable version of obs
#[0, 3, 0, 1, 4, 1, 0, 11, 8, 11, 17, 32, 32, 32, 49, 41, 46, 55, 39, 38, 34, 14, 16, 7, 0, 1, 2, 0, 1, 0]

# PMMH
# Tuning choices from preliminary runs
rw_cov = np.array([[0.38, -0.08, -0.07], [-0.08, 0.03, 0.08], [-0.08, 0.07, 1.96]])
theta0 = np.array([(0.18, -0.31, -4.60)], dtype=[('lnalpha', '<f8'), ('lnbeta', '<f8'), ('lngamma', '<f8')])
np.random.seed(3)
n_iterations = 50_000
thin = n_iterations // 2000
pmmh = mcmc.PMMH(ssm_cls=seir, prior=prior, data=obs, Nx=20, niter=n_iterations, verbose=100,
                 theta0=theta0, adaptive=False, rw_cov=rw_cov)
pmmh.run()

#Some plots

plt.ion()
# Pairs plot
import pandas as pd
import seaborn as sns
df = pd.DataFrame(pmmh.chain.theta)
print(f'Mean {df.mean()}')
print(f'Cov {df.cov()}')
sns.pairplot(df)

# traceplots
for p in ['lnalpha','lnbeta','lngamma']:  # loop over mu, theta, rho
    plt.figure()
    plt.plot(pmmh.chain.theta[p])
    plt.xlabel('iter')
    plt.ylabel(p)
  
#Prior vsposterior distributions    
prior_sample = prior.rvs(5000)
plt.style.use('ggplot')
plt.figure()
plt.hist(pmmh.chain.theta['lngamma'], density=True)
plt.hist(prior_sample['lngamma'], density=True, alpha=0.5)
plt.axvline(x=-2.6, color='r')
plt.xlabel('$log \\gamma$')

plt.figure()
plt.hist(pmmh.chain.theta['lnbeta'], density=True)
plt.hist(prior_sample['lnbeta'], density=True, alpha=0.5)
plt.axvline(x=-0.2, color='r')
plt.xlabel('$log \\beta$')

plt.figure()
plt.hist(pmmh.chain.theta['lnalpha'], density=True)
plt.hist(prior_sample['lnalpha'], density=True, alpha=0.5)
plt.axvline(x=0., color='r')
plt.xlabel('$log \\alpha$')

input('Press enter to continue') 

np.savetxt('seir_ref_sample.txt', pmmh.chain.theta[::thin])
