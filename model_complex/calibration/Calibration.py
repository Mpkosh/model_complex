import pymc as pm
import pandas as pd
import optuna
from sklearn.metrics import r2_score
import numpy as np
from scipy.optimize import dual_annealing

from ..models import BRModel

optuna.logging.set_verbosity(optuna.logging.ERROR)

class Calibration:

    def __init__(        
        self,
        init_infectious: list[int]|int,
        model: BRModel,
        data: pd.DataFrame,
        ) -> None:
        """
        Calibration class

        TODO

        :param init_infectious: Number of initial infected people  
        :param model: Model for calibration  
        :param data: Observed data for calibrating process  
        """
        self.rho = data['population_age_0-14'].iloc[-1] + data['population_age_15+'].iloc[-1]
        self.init_infectious = init_infectious
        self.model = model
        self.data = data


    def abc_calibration(self):
        """
        TODO

        """

        data, alpha_len, beta_len = self.model.params(self.data)

        def simulation_func(rng, alpha, beta, size=None):
            self.model.simulate(
                alpha=alpha, 
                beta=beta, 
                initial_infectious=self.init_infectious, 
                rho=round(self.rho/10), 
                modeling_duration=int(len(data)/alpha_len)
            )
            return self.model.newly_infected
        
        with pm.Model() as model:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1, shape=(alpha_len,))
            beta = pm.Uniform(name="beta", lower=0, upper=1, shape=(beta_len,))


            sim = pm.Simulator("sim", simulation_func, alpha, beta,
                            epsilon=3500, observed=data)
            
            idata = pm.sample_smc()

        posterior = idata.posterior.stack(samples=("draw", "chain"))


        alpha = [np.random.choice(posterior["alpha"][i], size=100) for i in range(alpha_len)]
        beta = [np.random.choice(posterior["beta"][i], size=100) for i in range(beta_len)]
        
        return alpha, beta, round(self.rho/10)
    


    def optuna_calibration(self, n_trials=1000):
        """
        TODO

        """

        data, alpha_len, beta_len = self.model.params(self.data)

        def model(trial):

            alpha = [trial.suggest_float(f'alpha_{i}', 0, 1) for i in range(alpha_len)]
            beta = [trial.suggest_float(f'beta_{i}', 0, 1) for i in range(beta_len)]


            self.model.simulate(
                alpha=alpha, 
                beta=beta, 
                initial_infectious=self.init_infectious, 
                rho=round(self.rho/10), 
                modeling_duration=int(len(data)/alpha_len)
            )

            return r2_score(data, self.model.newly_infected)


        study = optuna.create_study(direction="maximize")
        study.optimize(model, n_trials=n_trials)

        alpha = [study.best_params[f'alpha_{i}'] for i in range(alpha_len)]
        beta = [study.best_params[f'beta_{i}'] for i in range(beta_len)]

        return alpha, beta, round(self.rho/10)



    def annealing_calibration(self):
        """
        TODO

        """
        
        data, alpha_len, beta_len = self.model.params(self.data)

        lw = [0] * (alpha_len + beta_len)
        up = [1] * (alpha_len + beta_len)

        def model(x):
 
            alpha = x[:alpha_len]
            beta = x[alpha_len:]

            self.model.simulate(
                alpha=alpha, 
                beta=beta, 
                initial_infectious=self.init_infectious, 
                rho=round(self.rho/10), 
                modeling_duration=int(len(data)/alpha_len)
            )

            return -r2_score(data, self.model.newly_infected)
        
        ret = dual_annealing(model, bounds=list(zip(lw, up)))

        alpha = ret.x[:alpha_len]
        beta = ret.x[alpha_len:]

        return alpha, beta, round(self.rho/10)
    
    
    def mcmc_calibration(self, verbose=False, with_rho=False, with_initi=False, tune=2500, draws=500, chains=4):
        '''
        Parameters:
            - verbose -- show pymc progressbar
            - with_rho -- tune population size
            - with_initi -- tune initial infected
            - tune -- number of mcmc warmup samples
            - draws -- number of mcmc draws
            - chains -- number of chains
        '''
        
        init_inf = self.init_infectious
        rho = self.rho//10
        
        data, alpha_len, beta_len = self.model.params(self.data)
        progressbar = verbose
        data = np.nan_to_num(data).tolist() 

        def simulation_func(rng, alpha, beta, rho, init_inf, 
                            modeling_duration, size=None):
            
            self.model.simulate(
                alpha=np.array(alpha).flatten(), 
                beta=np.array(beta).flatten(), 
                initial_infectious=np.array(init_inf).flatten(), 
                rho=np.array(rho).flatten(), 
                modeling_duration=np.array(modeling_duration
                                          ).flatten()[0]
            )      
            return self.model.newly_infected
        
         
        
        with pm.Model() as pm_model:
            #alpha = pm.TruncatedNormal('a', mu=0.9, sigma=0.3, lower=0, upper=1, shape=alpha_len)
            #beta = pm.TruncatedNormal('b', mu=0.9, sigma=0.3, lower=0, upper=1, shape=beta_len)
            alpha = pm.Uniform(name="a", lower=0, upper=1, shape=alpha_len)
            beta = pm.Uniform(name="b", lower=0, upper=1, shape=beta_len)
            if with_rho:
                rho = pm.Uniform(name="rho", lower=50000, upper=5000000)
            if with_initi:
                init_inf = pm.Uniform(name="init_inf", lower=1, upper=1000, shape=alpha_len)

            # вынесем, тк для прогноза нужно будет задавать размер
            modeling_duration = [int(len(data)/alpha_len)] 
            sim = pm.Simulator("sim", simulation_func, alpha, beta,
                                rho, init_inf, modeling_duration,
                                epsilon=10000, 
                               ndims_params=[alpha_len,beta_len,1,alpha_len,1],
                               observed=data)
                
            # Differential evolution (DE) Metropolis sampler
            #step=pm.DEMetropolisZ(proposal_dist=pm.LaplaceProposal)
            step=pm.DEMetropolisZ()
            idata = pm.sample(tune=tune, draws=draws, 
                              cores=6, chains=chains,
                              step=step, progressbar=progressbar)
            idata.extend(pm.sample_posterior_predictive(idata,
                                                        progressbar=progressbar))

        return idata, data, simulation_func, pm_model
