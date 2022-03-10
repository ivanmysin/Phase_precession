import numpy as np
import pandas as pd
import time
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from scipy.signal.windows import parzen
import h5py
import neuron_simulation_lib as lib
from copy import copy
from multiprocessing import Pool
from default_params import get_neuron_structure
import presimulation_lib as plib
from default_params import default_param4optimization

def run_model(g_syn, sim_time, Erev, parzen_window, soma_idxes, dend_idxes, X = None):
    Erev = Erev + 60.0
    if X is None:
        g_syn_wcs = g_syn
    else:
        g_syn_wcs = np.copy(g_syn)
        W = X[0::3]
        C = X[1::3]
        S = X[2::3]
        # Weightings and centering
        for idx in range(W.size):
            g_syn_wcs[idx, :] *= W[idx] * np.exp(-0.5 * ((C[idx] - sim_time) / S[idx]) ** 2)

    neuron = get_neuron_structure()
    # print(neuron)

    neuron['compartments'][0]['soma']['input_conduntance'] = g_syn_wcs[soma_idxes, :]
    neuron['compartments'][0]['soma']['conduntances_Erev'] = Erev[soma_idxes]

    neuron['compartments'][1]['dendrite']['input_conduntance'] = g_syn_wcs[dend_idxes, :]
    neuron['compartments'][1]['dendrite']['conduntances_Erev'] = Erev[dend_idxes]

    # print(neuron)
    # print("################################################")


    dt = sim_time[1] - sim_time[0]
    duration = sim_time[-1]

    pyramidal = lib.ComplexNeuron(neuron["compartments"], neuron["connections"])
    pyramidal.integrate(dt, duration)


    Vsoma = pyramidal.getCompartmentByName('soma').getVhist()
    # Vdend = pyramidal.getCompartmentByName('dendrite').getVhist()
    spike_rate = np.zeros_like(Vsoma)


    peaks = find_peaks(Vsoma, height=30)[0]
    spike_rate[peaks] += 1
    spike_rate = np.convolve(spike_rate, parzen_window, mode='same')

    Vsoma = Vsoma - 60

    return spike_rate, Vsoma

##################################################################
def loss(X, teor_spike_rate, g_syn, sim_time, Erev, parzen_window, soma_idxes, dend_idxes):
    spike_rate, tmp = run_model(g_syn, sim_time, Erev, parzen_window, soma_idxes, dend_idxes, X)
    L = np.mean(np.log((teor_spike_rate + 1) / (spike_rate + 1))**2)
    return L


##################################################################
def optimization_model(num, param, data, output_path):
    ### Parameters for simulation
    duration = 3000  # ms
    dt = 0.1  # ms

    theta_freq = param['theta_freq']  # 8 Hz
    precession_slope = param['precession_slope']  # deg/cm
    animal_velosity = param['animal_velosity']  # cm/sec
    R_place_cell = param['R_place_cell']  # ray length
    place_field_center = 0.5 * duration  # center of similation
    sigma_place_field = param['sigma_place_field']  # cm
    ca3_center = place_field_center + 200  #
    ec3_center = place_field_center - 200  #


    data.loc["phi"] = np.deg2rad(data.loc["phi"])
    data.loc["kappa"] = [plib.r2kappa(r) for r in data.loc["R"]]

    soma_idxes, dend_idxes = plib.get_soma_dend_idxes(data)
    ###################################################################
    ### Делаем предвычисления
    sim_time = np.arange(0, duration, dt)
    sim_time = np.around(sim_time, 2)
    precession_slope = animal_velosity * np.deg2rad(precession_slope)
    kappa_place_cell = plib.r2kappa(R_place_cell)
    sigma_place_field = 1000 * sigma_place_field / animal_velosity  # recalculate to ms

    teor_spike_rate = plib.get_teor_spike_rate(sim_time, precession_slope, theta_freq, kappa_place_cell,
                                          sigma=sigma_place_field, center=place_field_center)

    parzen_window = parzen(1001)  # parzen(401) # parzen(701)
    ####################################################################
    print("integrate synaptic coductances")
    g_syn, Erev = plib.get_gsyns(data, theta_freq, sim_time)

    n_pops = len(data.columns)
    W = 0.1 * np.ones(n_pops, dtype=np.float64)
    W[0] = 0.5
    W[1] = 0.5

    centers = np.zeros_like(W) + place_field_center
    centers[0] = ca3_center
    centers[1] = ec3_center

    sigmas = np.zeros_like(centers) + sigma_place_field

    if param["use_x0"]:
        X = np.zeros(n_pops * 3, dtype=np.float64)
        X[0::3] = W
        X[1::3] = centers
        X[2::3] = sigmas
    else:
        X = None

    bounds = []
    for bnd_idx in range(n_pops * 3):
        if bnd_idx % 3 == 0:
            bounds.append([0, 1])
        elif bnd_idx % 3 == 1:
            bounds.append([0, duration])
        elif bnd_idx % 3 == 2:
            bounds.append([100, 15000])

    # Изменяем границы для параметров для СА3
    bounds[0][0] = 0.01  # вес не менее
    bounds[1][0] = place_field_center  # центр входа от СА3 не ранее центра в СА1

    # Изменяем границы для параметров для EC3
    bounds[3][0] = 0.01  # вес не менее
    bounds[4][1] = place_field_center  # центр входа от EC3 ранее центра в СА1

    loss_args = (teor_spike_rate, g_syn, sim_time, Erev, parzen_window, soma_idxes, dend_idxes)

    timer = time.time()
    print('starting optimization ... ')
    sol = differential_evolution(loss, x0=X, popsize=24, atol=1e-3, recombination=0.7, \
                                 mutation=0.2, args=loss_args, bounds=bounds, maxiter=3, \
                                 workers=-1, updating='deferred', disp=True, \
                                 strategy='best2bin')
    X = sol.x

    print("Time of optimization ", time.time() - timer, " sec")
    print("success ", sol.success)
    print("message ", sol.message)
    print("number of interation ", sol.nit)

    run_model_with_parameters([X, g_syn, Erev, param, sim_time, output_path, num, teor_spike_rate, soma_idxes, dend_idxes])

    return


def run_model_with_parameters(args):
    X, g_syn, Erev, param, sim_time, output_path, filename, teor_spike_rate, soma_idxes, dend_idxes = args
    parzen_window = parzen(15)
    place_field_center = sim_time[sim_time.size//2]

    spike_rate, Vhist = run_model(g_syn, sim_time, Erev, parzen_window, soma_idxes, dend_idxes, X=X)
    W = X[0::3]
    C = X[1::3]
    S = X[2::3]

    with h5py.File(output_path + f'{filename}.hdf5', "w") as hdf_file:
        hdf_file.create_dataset('V', data=Vhist)
        hdf_file.create_dataset('spike_rate', data=spike_rate)
        hdf_file.create_dataset('teor_spike_rate', data=teor_spike_rate)
        hdf_file.create_dataset('Weights', data=W)
        hdf_file.create_dataset('Centers', data=(C - place_field_center))
        hdf_file.create_dataset('Sigmas', data=S)
        for name, value in param.items():
            hdf_file.attrs[name] = value




def multipal_run(output_path, X, params_list, data):
    # run optimizeed model with different params
    dt = 0.1
    duration = 10000
    sim_time = np.arange(0, duration, dt)
    sim_time = np.around(sim_time, 2)

    run_model_args = []


    for param_idx, param in enumerate(params_list):
        filename = str(param_idx)
        teor_spike_rate = np.empty(0)
        theta_freq = param['theta_freq']
        g_syn, Erev = plib.get_gsyns(data, theta_freq, sim_time)
        run_model_args.append((X, g_syn, Erev, param, sim_time, output_path, filename, teor_spike_rate))

    with Pool(processes=4) as p:
        p.map(run_model_with_parameters, run_model_args)









