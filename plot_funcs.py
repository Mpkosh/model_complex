import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm, trange
from model_complex import Calibration, EpidData, FactoryBRModel
from timeit import default_timer as timer
import pymc as pm
import arviz as az


def pred_plot(year, data, data_mtest, 
              pp, idata, with_ylim=False):
    
    alpha_len = idata.posterior.a.shape[-1]
    train_size = len(data)//alpha_len
    test_size = len(data_mtest)//alpha_len
    
    train_steps = np.arange(train_size)
    test_steps = np.arange(train_size+test_size)
    
    # 'sim_forecast' chain: 4 draw: 500 sim_forecast_dim_2: ...
    values = pp.predictions["sim_forecast"]
    sim_value = idata.posterior_predictive.sim
     
    
    fig, ax = plt.subplots(1,alpha_len, sharex=True, 
                           sharey=True, figsize=(9,5))
    # чтобы работало и при одном ax
    ax = np.array([ax]).flatten()
    
    # для каждой возрастной группы
    for i in range(alpha_len):
        model_time = [train_size]
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]  
        
        # _____calibration plot part
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        # ci
        ci = .95
        ax[i].fill_between(x=train_steps,
                         y1=sim_part.quantile(q=round(0.5 - ci/2, 3), dim=['chain', 'draw']), 
                         y2=sim_part.quantile(q=round(0.5 + ci/2, 3), dim=['chain', 'draw']),
                         color='lightgray', alpha=.5)
        # median calibration line
        ax[i].plot(train_steps, sim_part.quantile(q=.5, dim=['chain', 'draw']), color='lightgray', lw=2)
        # all lines
        calib_lines = np.array(sim_part).reshape(-1,train_size).T
        ax[i].plot(train_steps, calib_lines, color='gray', alpha=0.005) 
        
        # _____prediction plot part
        
        model_time = [train_size+test_size]
        #data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]] 
        values_part = values[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        preds_lower = values_part.quantile(q=round(0.5 - ci/2, 3), 
                                           dim=['chain', 'draw'])
        pred_higher = values_part.quantile(q=round(0.5 + ci/2, 3), 
                                           dim=['chain', 'draw'])
        # ci
        ax[i].fill_between(x=test_steps[train_size:],
                         y1=preds_lower[train_size:], 
                         y2=pred_higher[train_size:],
                         color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')
        # predictive lines
        preds = np.array(values_part).reshape(-1,train_size+test_size).T
        ax[i].plot(test_steps[train_size:], 
                 preds[train_size:], 
                 color='blue', alpha=0.005);    
        # median predictive line
        median_pred = np.array(values_part.quantile(q=.5, 
                                                    dim=['chain', 'draw']))
        ax[i].plot(test_steps[train_size:], 
                 median_pred[train_size:], 
                 color='gray', lw=2, label=f'median')

        ax[i].axvline(train_size-0.5, ls="--", color='gray', lw=1)


        # data
        ax[i].plot(train_steps, 
                   data[i*train_size:i*train_size+train_size], 
                   "o", ls='', color='brown', 
                   markeredgecolor='white', label='train data')
        ax[i].plot(test_steps[train_size:], 
                   data_mtest[i*test_size:i*test_size+test_size],
                   "o", ls='', color='forestgreen', 
                   markeredgecolor='white', label='test data')

        if with_ylim:
            high_lim = np.max([median_pred.max()*1.25,
                              np.max(data[i*train_size:i*train_size+train_size
                                  ])*1.25,
                              np.max(data_mtest[i*test_size:i*test_size+test_size
                                        ])*1.25])
            ax[i].set_ylim(None, high_lim)
            
        ax[i].set_xlabel('День')
        ax[i].set_ylabel('Новые заболевшие (чел.)')
        ax[i].grid()
        ax[i].set_title(f'{year}-{year+1} г.г.')
        ax[i].legend();

    
def pred_plot_ax(ax, year, data, data_mtest, 
              pp, idata, with_ylim=False, time=0):
    
    alpha_len = idata.posterior.a.shape[-1]
    train_size = len(data)//alpha_len
    test_size = len(data_mtest)//alpha_len
    
    train_steps = np.arange(train_size)
    test_steps = np.arange(train_size+test_size)
    
    # 'sim_forecast' chain: 4 draw: 500 sim_forecast_dim_2: ...
    values = pp.predictions["sim_forecast"]
    sim_value = idata.posterior_predictive.sim
    
    col = 0
    if year >= 2015:
        col = 2
        
    # для каждой возрастной группы
    for i in range(alpha_len):
        model_time = [train_size]
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]  
        
        # _____calibration plot part
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]    
        # ci
        ci = .95
        ax[col+i].fill_between(x=train_steps,
                         y1=sim_part.quantile(q=round(0.5 - ci/2, 3), dim=['chain', 'draw']), 
                         y2=sim_part.quantile(q=round(0.5 + ci/2, 3), dim=['chain', 'draw']),
                         color='lightgray', alpha=.5)
        # median calibration line
        ax[col+i].plot(train_steps, sim_part.quantile(q=.5, dim=['chain', 'draw']), color='lightgray', lw=2)
        # all lines
        calib_lines = np.array(sim_part).reshape(-1,train_size).T
        ax[col+i].plot(train_steps, calib_lines, color='gray', alpha=0.005) 
        
        # _____prediction plot part
        
        model_time = [train_size+test_size]
        #data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]] 
        values_part = values[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        preds_lower = values_part.quantile(q=round(0.5 - ci/2, 3), 
                                           dim=['chain', 'draw'])
        pred_higher = values_part.quantile(q=round(0.5 + ci/2, 3), 
                                           dim=['chain', 'draw'])
        # ci
        ax[col+i].fill_between(x=test_steps[train_size:],
                         y1=preds_lower[train_size:], 
                         y2=pred_higher[train_size:],
                         color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')
        # predictive lines
        preds = np.array(values_part).reshape(-1,train_size+test_size).T
        ax[col+i].plot(test_steps[train_size:], 
                 preds[train_size:], 
                 color='blue', alpha=0.005);    
        # median predictive line
        median_pred = np.array(values_part.quantile(q=.5, 
                                                    dim=['chain', 'draw']))
        ax[col+i].plot(test_steps[train_size:], 
                 median_pred[train_size:], 
                 color='gray', lw=2, label=f'median')

        ax[col+i].axvline(train_size-0.5, ls="--", color='gray', lw=1)


        # data
        ax[col+i].plot(train_steps, 
                   data[i*train_size:i*train_size+train_size], 
                   "o", ls='', color='brown', 
                   markeredgecolor='white', label='train data')
        ax[col+i].plot(test_steps[train_size:], 
                   data_mtest[i*test_size:i*test_size+test_size],
                   "o", ls='', color='forestgreen', 
                   markeredgecolor='white', label='test data')

        if with_ylim:
            high_lim = np.max([median_pred.max()*1.25,
                              np.max(data[i*train_size:i*train_size+train_size
                                  ])*1.25,
                              np.max(data_mtest[i*test_size:i*test_size+test_size
                                        ])*1.25])
            ax[col+i].set_ylim(None, high_lim)
            
        ax[col+i].text(x=0.01,y=0.85, s=f'Time = {time:.3f}', 
            ha='left', va='top', transform=ax[col+i].transAxes)    
        ax[col+i].set_xlabel('День')
        ax[col+i].set_ylabel('Новые заболевшие (чел.)')
        ax[col+i].grid()
        ax[col+i].set_title(f'{year}-{year+1} г.г.');
    
    

# _____________________ FOR CALIBRATION    

def plots(idata, data, year, simulation_func, cal_model,
          with_trace=True, with_rho=False, with_initi=False,
          show_values=True, for_pred=False):
    
    sim_value = idata.posterior_predictive.sim
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    
    alpha_len = idata.posterior.a.shape[-1]    
    model_time = [len(data)//alpha_len]
    x = np.arange(model_time[0])
    
    fig, ax = plt.subplots(1,alpha_len, sharex=True, 
                           sharey=True, figsize=(9,5))
    # чтобы работало и при одном ax
    ax = np.array([ax]).flatten()
    # для каждой возрастной группы
    for i in range(alpha_len):
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        r2_part = r2_score(data_part[len(data_part)//2:], 
                             sim_part.quantile(q=.5, 
                                               dim=['chain', 'draw']
                                              )[sim_part.shape[-1]//2:])
        
        # posterior predictive lines
        ax[i].plot(np.array(sim_part).reshape(-1,len(data_part)).T, 
                 color='blue', alpha=0.005) 
        # ppc median
        ax[i].plot(sim_part.quantile(q=.5, dim=['chain', 'draw']),
                 color='lightgray', lw=2,
                 label=f'median (R^2 = {r2_part:.3f})')

        # real data
        ax[i].plot(data_part, "o", ls='', color='brown', 
                   markeredgecolor='white', label='real data')

        ci = .95
        # posterior predictive ci
        ax[i].fill_between(x=x,
                         y1=sim_part.quantile(q=round(0.5 - ci/2, 3), 
                                              dim=['chain', 'draw']), 
                         y2=sim_part.quantile(q=round(0.5 + ci/2, 3), 
                                              dim=['chain', 'draw']),
                         color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%')

        ax[i].set_xlabel('День')
        ax[i].set_ylabel('Новые заболевшие (чел.)')
        ax[i].grid()
        ax[i].set_title(f'{year}-{year+1} г.г.')
        ax[i].legend();
    
    if with_trace:
        pm.plot_trace(idata);
        az.plot_posterior(idata);
        return az.summary(idata)
    
    
def fig_for_subplot(ax, idata, data, year, simulation_func,
                    cal_model, with_rho=False, 
                    with_initi=False, time=0):
    col = 0
    if year >= 2015:
        col = 2
    
    sim_value = idata.posterior_predictive.sim
    
    alpha_len = idata.posterior.a.shape[-1]    
    model_time = [len(data)//alpha_len]
    x = np.arange(model_time[0])

    
    # для каждой возрастной группы
    for i in range(alpha_len):
        
        data_part = data[i*model_time[0]:i*model_time[0]+model_time[0]]
        sim_part = sim_value[:,
                             :,
                             i*model_time[0]:i*model_time[0]+model_time[0]]
        r2_part = r2_score(data_part[len(data_part)//2:], 
                             sim_part.quantile(q=.5, 
                                               dim=['chain', 'draw']
                                              )[sim_part.shape[-1]//2:])
        ci = .95
        # posterior predictive ci
        ax[col+i].fill_between(x=x,
             y1=sim_part.quantile(q=round(0.5 - ci/2, 3), 
                                  dim=['chain', 'draw']), 
             y2=sim_part.quantile(q=round(0.5 + ci/2, 3), 
                                  dim=['chain', 'draw']),
             color='skyblue', alpha=.5, label=f'CI {ci*100:.0f}%'
                              )
        
        # calibration lines
        ax[col+i].plot(np.array(sim_part).reshape(-1,len(data_part)).T, 
                 color='blue', alpha=0.005) 
        #  median
        ax[col+i].plot(sim_part.quantile(q=.5, dim=['chain', 'draw']),
                 color='lightgray', lw=2,
                 label=f'median (R^2 = {r2_part:.3f})')

        # real data
        ax[col+i].plot(data_part, "o", ls='', color='brown', 
                   markeredgecolor='white', label='real data')
        
        ax[col+i].text(x=0.01,y=0.95, s=f'R\N{SUPERSCRIPT TWO}(median) = {r2_part:.3f}', 
                ha='left', va='top', transform=ax[col+i].transAxes)
        ax[col+i].text(x=0.01,y=0.85, s=f'Time = {time:.3f}', 
                ha='left', va='top', transform=ax[col+i].transAxes)
        ax[col+i].set_title(f'{year}-{year+1} г.г., age: {i}')
        ax[col+i].grid()



def grid_plots(with_rho=True, with_initi=True, weeks_bfr = 3,
               years = [2010,2011,2012,2013,2014,2015,
                        2016,2017,2018,2019],
              tune=2500, draws=500, chains=4):
    fig, ax = plt.subplots(5,2, sharex=True, sharey=True, figsize=(9,11))
    init_infect = [100]

    for idx, year in enumerate(tqdm(years)):
        data_orig = EpidData('spb', './', f'7-01-{year}', f'6-20-{year+1}')
        peak_idx = data_orig.reset_index(drop=True)['total'].nlargest(1).index[0]
        data_train, data_test = data_orig.iloc[:peak_idx-weeks_bfr+1], \
                                    data_orig.iloc[peak_idx-weeks_bfr+1:]

        model = TotalBRModel()

        d = Calibration(init_infect, model, data_train)
        start = timer()
        idata, data, simulation_func, \
            pm_model,rho = d.mcmc_calibration(with_rho=with_rho,
                                              with_initi=with_initi,
                                              tune=tune, draws=draws, chains=chains)
        pp = predictions(simulation_func, data, idata, duration=data_orig.shape[0], 
                         rho=rho, alpha_len=d.model.alpha_len,
                         with_rho=with_rho, with_initi=with_initi)  
        end = timer()
        
        col = 0
        if year >= 2015:
            col = 1
            idx -= 5

        pred_plot_ax(ax[idx, col], year, data_train, data_test,
                     pp, idata, end-start, with_ylim=True)

    fig.supxlabel('День')   
    fig.supylabel('Новые заболевшие (чел.)')

    handles, labels = ax[-1,0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 1));
    
    
def predictions(simulation_func, idata, duration, 
                cal_model, with_rho=True, with_initi=True):
    
    alpha_len = cal_model.model.alpha_len
    beta_len = cal_model.model.beta_len
    
    with pm.Model() as forecast_m:
        alpha = pm.Uniform(name="a", shape=alpha_len)
        beta = pm.Uniform(name="b", shape=beta_len)
        rho = cal_model.rho//10
        if with_rho:
            rho = pm.Uniform(name="rho")
        init_inf = cal_model.init_infectious
        if with_initi:
            init_inf = pm.Uniform(name="init_inf", shape=alpha_len)

        pm.Simulator("sim_forecast", simulation_func, alpha, beta,
                 rho, init_inf, [duration], epsilon=10000,
                     ndims_params=[alpha_len,beta_len,
                                   1,alpha_len,1])
        pp = pm.sample_posterior_predictive(idata, 
                                            var_names=["sim_forecast"],
                                            predictions=True,
                                            progressbar=False)
    return pp