#
# montecarlo.py
#
# Definition of general Monte Carlo solver class
#

import numpy as np

class MonteCarlo:
    
    def __init__(self, domain = np.array([[0,1]]), aggregate_func = lambda x : np.mean(x)):
        self.domain = domain
        self.aggregate_func = aggregate_func
    
    def simulate(self, n = 100):
        
        # Generate samples
        samples = np.random.rand(n, self.domain.shape[0])
        
        # Scale samples to problem domain
        domain_n = np.tile(self.domain[:,:], (n,1,1))
        samples = samples * np.abs(domain_n[:,:,1] - domain_n[:,:,0])
        samples = samples + domain_n[:,:,0]
        
        # Aggregate results
        return self.aggregate_func(samples)