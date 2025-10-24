import numpy as np

np.random.seed(1)
ref_sample = np.random.randn(2000)
ref_sample[0:1000] *= 0.1

if ref_sample.min() < -10 or ref_sample.max() > 10:
    raise ValueError("Samples outside prior support")

np.savetxt('gaussian_mixture_ref_sample.txt', ref_sample)