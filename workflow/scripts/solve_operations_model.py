import numpy as np
import pypsa
import pandas as pd
import logging
from tqdm import tqdm
from joblib import Parallel, delayed


def solve_operations_model(n, commit=False, load_shedding = False, snapshots=None, solver_name='gurobi'):
    # Optimize the network for the specified time period
    if load_shedding: n.optimize.add_load_shedding(sign=1, marginal_cost=10000,suffix=' load')
    n.optimize(snapshots, solver_name=solver_name)
    revenue = n.statistics.revenue()
    revenue.index = revenue.index.map(lambda x: '_'.join(x[:2]))
    return n, revenue


def update_network_data(n, sample):
    """Update the network data with the given sample
    """
    n_loads = n.loads_t.p_set.shape[1]
    n_generators = n.generators_t.p_max_pu.shape[1]
    # update data
    n.loads_t.p_set.iloc[:,:] = sample[:,:n_loads]
    n.generators_t.p_max_pu.iloc[:,:] = sample[:,n_loads:n_loads+n_generators]

    ## Issue with the p_min_pu < p_max_pu. Clipping max values. Could also set these to 0 to allow curtailment.
    n.generators_t.p_min_pu = n.generators_t.p_min_pu.clip(0, n.generators_t.p_max_pu)
    return n

def update_datasets(dk, dleft, idxs):
    '''
    helper function to update datasets
    function takes index, idx, and moves it from dleft to dk'''
    if dk is None:
        dk = dleft[idxs]
    else:
        dk = np.vstack([dk, dleft[idxs]])
    dleft = np.delete(dleft, idxs, axis=0)
    return dk, dleft

def iter_samples(GP, method, X, Xleft, fleft, loops=100):
    for i in tqdm(range(loops)):
        # get sample from encoded data
        xk_idx = GP.get_next_sample(Xleft, next_sample_method=method)
        xk, Xleft = update_datasets(None, Xleft, xk_idx)
        fk, fleft = update_datasets(None, fleft.reshape(-1,1), xk_idx)
        GP.add_data(xk, fk)
    return GP.evaluate(X)

def process_iteration(n, samples, i):
    # update network data
    n = update_network_data(n, samples[i, :, :])
    
    # solve operations model
    n_hours = snakemake.config['solving']['options']['nhours']
    n, revenue = solve_operations_model(n, commit=False, load_shedding=False, snapshots=n.snapshots[:n_hours], solver_name=solver)
    return i, revenue

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    #Configs
    solver = snakemake.config['solving']['solver']['name']
    stochastic_operations = snakemake.config['solving']['operations']['stochastic']

    # Load Data
    n = pypsa.Network(snakemake.input.network)


    if stochastic_operations:
        samples = np.load(snakemake.input.samples)
        #Run Operations Model
        operations_results = pd.DataFrame(np.zeros(shape=(len(samples),2)), columns=['sample_num', 'production_cost'])

        num_iterations = snakemake.config['solving']['operations']['iterations']
        results = Parallel(n_jobs=-1)(delayed(process_iteration)(n, samples, i) for i in tqdm(range(num_iterations)))

        # Assuming operations_results is a DataFrame
        operations_results = pd.DataFrame(results, columns=['iteration', 'production_cost'])

        # Sort the DataFrame by iteration if needed
        operations_results.sort_values('iteration', inplace=True)
        operations_results.to_csv(snakemake.output.operations_results, index=False)
    else:
        #Run Operations Model
        n_hours = snakemake.config['solving']['options']['nhours']
        n, revenue = solve_operations_model(n, commit=False, 
                                                             load_shedding=False, 
                                                             snapshots=n.snapshots[:n_hours], 
                                                             solver_name=solver
                                                             )
        revenue.to_csv(snakemake.output.production_cost)
        n.export_to_netcdf(snakemake.output.network_solved)