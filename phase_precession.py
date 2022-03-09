import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import parzen
import time
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from numba import jit
import h5py
import lib
from scipy.signal import find_peaks
from numpy.random import randint
from copy import copy
import os
from scipy import signal as sig
from multiprocessing import Pool

def run_model(g_syn, sim_time, Erev, parzen_window, issaveV):
    Erev = Erev + 60.0
    soma_idxes = np.array( [0, 2, 3, 7, 8 ] ).astype(np.int16)
    dend_idxes = np.array( [ 1, 4, 5, 6 ] ).astype(np.int16)
    g_syn = g_syn.T


    # print(g_syn[soma_idxes, :].shape)
    # print(Erev[soma_idxes].shape)

    soma_params = {
        "V0": -5.0,
        "C": 3.0,
        "Iextmean": 0.0,
        "Iextvarience": 0.0001,
        "ENa": 120.0,
        "EK": -15.0,
        "El": -5.0,
        "ECa": 140.0,
        "CCa": 0.05,
        "sfica": 0.13,
        "sbetaca": 0.075,
        "gbarNa": 30.0,
        "gbarK_DR": 17.0,
        "gbarK_AHP": 0.8,
        "gbarK_C ": 15.0,
        "gl": 0.1,
        "gbarCa": 6.0,

        "input_conduntance": g_syn[soma_idxes, :], # np.empty((0, 0), dtype=np.float32),
        "conduntances_Erev": Erev[soma_idxes], # np.empty(0, dtype=np.float32),
    }

    dendrite_params = {
        "V0": -5.0,
        "C": 3.0,
        "Iextmean": 0.0,
        "Iextvarience": 0.0001,
        "ENa": 120.0,
        "EK": -15.0,
        "El": -5.0,
        "ECa": 140.0,
        "CCa": 0.05,
        "sfica": 0.13,
        "sbetaca": 0.075,
        "gbarNa": 0.0,
        "gbarK_DR": 0.0,
        "gbarK_AHP": 0.8,
        "gbarK_C ": 5.0,
        "gl": 0.1,
        "gbarCa": 5.0,

        "input_conduntance": g_syn[dend_idxes, :], # np.empty((0, 0), dtype=np.float32),  # np.zeros( (1, 10000), dtype=np.float32) + 0.05, #
        "conduntances_Erev": Erev[dend_idxes], # np.empty(0, dtype=np.float32),  # np.zeros(1, dtype=np.float32) + 120, #
    }

    connection_params = {
        "compartment1": "soma",
        "compartment2": "dendrite",
        "p": 0.5,
        "g": 1.5,
    }

    soma = {"soma": soma_params.copy()}
    # soma["soma"]["V0"] = 0.5 * np.random.randn()

    dendrite = {"dendrite": dendrite_params.copy()}
    dendrite["dendrite"]["V0"] = soma["soma"]["V0"]

    connection = connection_params.copy()
    neuron = {
        "type": "pyramide",
        "compartments": [soma, dendrite],
        "connections": [connection, ]
    }
    dt = sim_time[1] - sim_time[0]
    duration = sim_time[-1]

    # timer = time.time()
    pyramidal = lib.ComplexNeuron(neuron["compartments"], neuron["connections"])
    # print('Creation object time ', time.time() - timer, ' sec')
    pyramidal.integrate(dt, duration)
    # print('Simulation time ', time.time() - timer, ' sec')

    Vsoma = pyramidal.getCompartmentByName('soma').getVhist()
    # Vdend = pyramidal.getCompartmentByName('dendrite').getVhist()
    spike_rate = np.zeros_like(Vsoma)

    # print(sim_time.size, Vsoma.size, duration)

    peaks = find_peaks(Vsoma, height=30)[0]
    spike_rate[peaks] += 1
    spike_rate = np.convolve(spike_rate, parzen_window, mode='same')

    Vsoma = Vsoma - 60

    return spike_rate, Vsoma









#@jit(nopython=True)
def loss(X, teor_spike_rate, g_syn, sim_time, Erev, parzen_window, issaveV):


    n_pops = X.size//3

    g_syn_wcs = np.copy(g_syn)
    W = X[0::3]
    C = X[1::3]
    S = X[2::3]

    # взвешивание и центрирование гауссианой
    for idx in range(n_pops):
        g_syn_wcs[:, idx] *= W[idx] *  np.exp(-0.5 * ((C[idx] - sim_time) / S[idx])**2)


    spike_rate, tmp = run_model(g_syn_wcs, sim_time, Erev, parzen_window, issaveV)
    L = np.mean( np.log( (teor_spike_rate+1)/(spike_rate+1) )**2 )
    return L
##################################################################
def main(num, param):
    print('start optimization')
    ###################################################################
    ### Параметры для симуляции
    duration = 3000 # ms
    dt = 0.1        # ms
    
    theta_freq = param['theta_freq'] # 8 Hz
    precession_slope = param['precession_slope']  # deg/cm
    animal_velosity = param['animal_velosity']  # cm/sec
    R_place_cell = param['R_place_cell']  # ray length
    place_field_center = 0.5*duration # center of similation
    sigma_place_field = param['sigma_place_field'] # cm 
    ca3_center = place_field_center + 200 #
    ec3_center = place_field_center - 200 #
    
    output_path = "./output/"
    datafile = "inputs_data_test.csv"
    conductance_file = output_path + "conductances.hdf5"
    ###################################################################
    
    ### Делаем предвычисления
    sim_time = np.arange(0, duration, dt)
    sim_time = np.around(sim_time, 2)
    precession_slope = animal_velosity * np.deg2rad(precession_slope)
    kappa_place_cell = r2kappa(R_place_cell)
    sigma_place_field = 1000 * sigma_place_field / animal_velosity # recalculate to ms
    
    teor_spike_rate = get_teor_spike_rate(sim_time, precession_slope, theta_freq, kappa_place_cell,  sigma=sigma_place_field, center=place_field_center)


    data = pd.read_csv(datafile, header=0, comment="#", index_col=0)
    data.loc["phi"]  = np.deg2rad(data.loc["phi"])
    data.loc["kappa"] = [ r2kappa(r) for r in data.loc["R"] ]
    parzen_window = parzen(1001) # parzen(401) # parzen(701)
    ####################################################################
    g_syn = np.zeros((sim_time.size, len(data.columns)), dtype=np.float64)
    Erev = np.zeros( len(data.columns), dtype=np.float64)
    # inegrate_g(t, z, tau_rise, tau_decay, mu, kappa, freq)
    print("integrate synaptic coductances")

    with h5py.File(conductance_file, "w") as hdf_file:
        hdf_file.attrs["dt"] = dt
        hdf_file.attrs["duration"] = duration
        for inp_idx, input_name in enumerate(data.columns):
            args = (data.loc["tau_rise"][input_name], data.loc["tau_decay"][input_name], data.loc["phi"][input_name], data.loc["kappa"][input_name], theta_freq)
            sol = solve_ivp(inegrate_g, t_span=[0, duration], t_eval=sim_time, y0=[0, 0], args=args, dense_output=True)
            # g = sol.sol(sim_time)[0]
            g = sol.y[0]
            g *= 1.0 / np.max(g) #0.1 for LIF !!!!!!!!
            g_syn[:, inp_idx] = g
            Erev[inp_idx] = data.loc["E"][input_name]
    
            hdf_file.create_dataset(input_name, data=g)

    # with h5py.File(conductance_file, "r") as hdf_file:
    #      for inp_idx, input_name in enumerate(data.columns):
    #         g_syn[:, inp_idx] = hdf_file[input_name][:]
    #         Erev[inp_idx] = data.loc["E"][input_name]

    n_pops = len(data.columns)
    
    W = 0.1*np.ones(n_pops, dtype=np.float64)
    W[0] = 0.5
    W[1] = 0.5

    centers = np.zeros_like(W) + place_field_center
    centers[0] = ca3_center
    centers[1] = ec3_center

    # centers[2] += -200.0 # cck
    # centers[3] += 200.0 # pv
    # centers[4] += 200.0 # ngf
    # centers[5] += -200.0 # ivy
    # centers[6] += 400.0 # olm
    # centers[7] += 0 # aac
    # centers[8] += 0 # bis

    sigmas = np.zeros_like(centers) + sigma_place_field

    if param["use_x0"]:
        X = np.zeros(n_pops * 3, dtype=np.float64)
        X[0::3] = W
        X[1::3] = centers
        X[2::3] = sigmas
    else:
        X = None

    print("start simulation")
    
    bounds = []
    for bnd_idx in range(n_pops * 3):
        if bnd_idx%3 == 0:
            bounds.append([0, 1])
        elif bnd_idx%3 == 1:
            bounds.append([0, duration])
        elif bnd_idx%3 == 2:
            bounds.append([100, 15000])
    
    # Изменяем границы для параметров для СА3
    bounds[0][0] = 0.01 # вес не менее
    bounds[1][0] = place_field_center # центр входа от СА3 не ранее центра в СА1
    
    # Изменяем границы для параметров для EC3
    bounds[3][0] = 0.01 # вес не менее
    bounds[4][1] = place_field_center # центр входа от EC3 ранее центра в СА1
    
   
    
    loss_args = (teor_spike_rate, g_syn, sim_time, Erev, parzen_window, False)

    timer = time.time()
    # recombination=1.7 # 0.5
    # mutation=1.2,  # 0.1 - 0.2
    sol = differential_evolution(loss, x0=X, popsize=24, atol=1e-3, recombination=0.7, \
                                 mutation=0.2, args=loss_args, bounds=bounds,  maxiter=100, \
                                 workers=-1, updating='deferred', disp=True, \
                                 strategy='best2bin')

    X = sol.x
    
    print("Time of optimization ", time.time() - timer, " sec")
    print("success ", sol.success)
    print("message ", sol.message)
    print("number of interation ", sol.nit)
    
    g_syn_wcs = np.copy(g_syn)
    W = X[0::3]
    C = X[1::3]
    S = X[2::3]
    

    
    # взвешивание и центрирование гауссианой
    for idx in range(n_pops):
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ((C[idx] - sim_time) / S[idx])**2)
    
    spike_rate, Vhist = run_model(g_syn_wcs, sim_time, Erev, parzen_window, True)
    firings = 0.001 * sig.find_peaks(Vhist, height=-10)[0] * dt

    xpos = firings/animal_velosity
    theta_phases = 2*np.pi*theta_freq*firings
    theta_phases = theta_phases % (2*np.pi)
    theta_phases[theta_phases > np.pi] -= 2*np.pi
    
    fig, axes = plt.subplots(nrows=3)
    axes[0].plot(sim_time, Vhist)
    axes[1].plot(sim_time, teor_spike_rate,  linewidth=1, label='target spike rate')
    axes[1].plot(sim_time, spike_rate, linewidth=1, label='simulated spike rate')
    axes[1].legend(loc='upper left')

    axes[2].scatter(xpos, theta_phases)
    fig.savefig(output_path + f"spike_rate_{num}.png")
    
    
    with h5py.File(output_path + f'{num}.hdf5', "w") as hdf_file:
        hdf_file.create_dataset('V', data=Vhist)
        hdf_file.create_dataset('spike_rate', data=spike_rate)
        hdf_file.create_dataset('teor_spike_rate', data=teor_spike_rate)
        hdf_file.create_dataset('Weights', data=W)
        hdf_file.create_dataset('Centers', data=(C - place_field_center) )
        hdf_file.create_dataset('Sigmas', data=S)
        for name, value in param.items():
            hdf_file.attrs[name] = value
    
    plt.close('all')
    return 
            
def run_model_with_parameters(args):
    params, default_param, W, C, S, dt, duration, output_path, filename = args

    sim_time = np.arange(0, duration, dt)

    C = C * default_param['animal_velosity']/params['animal_velosity'] + 0.5*duration
    S = S * default_param['animal_velosity']/params['animal_velosity']

    datafile = "inputs_data.csv"
    data = pd.read_csv(datafile, header=0, comment="#", index_col=0)
    data.loc["phi"]  = np.deg2rad(data.loc["phi"])
    data.loc["kappa"] = [ r2kappa(r) for r in data.loc["R"] ]
    parzen_window = parzen(15)
    g_syn = np.zeros((sim_time.size, len(data.columns)), dtype=np.float64)
    Erev = np.zeros( len(data.columns), dtype=np.float64)

    theta_freq = params['theta_freq']

    # взвешивание и центрирование гауссианой
    for inp_idx, input_name in enumerate(data.columns):
        args = (data.loc["tau_rise"][input_name], data.loc["tau_decay"][input_name], data.loc["phi"][input_name], data.loc["kappa"][input_name], theta_freq)
        sol = solve_ivp(inegrate_g, t_span=[0, duration], t_eval=sim_time, y0=[0, 0], args=args, dense_output=True)
        #g = sol.sol(sim_time)[0]
        g = sol.y[0]
        g *= 1.0 / np.max(g)
        g_syn[:, inp_idx] = g
        Erev[inp_idx] = data.loc["E"][input_name]

    g_syn_wcs = g_syn
    for idx in range(W.size):
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ((C[idx] - sim_time) / S[idx])**2)

    spike_rate, Vhist = run_model(g_syn_wcs, sim_time, Erev, parzen_window, True)

    # fig, axes = plt.subplots(nrows=2, sharex=True)
    # axes[0].plot(sim_time, Vhist)
    # axes[1].plot(sim_time, spike_rate, linewidth=1, label='simulated spike rate')
    #
    # axes[1].legend(loc='upper left')
    # fig.savefig(output_path + f"spike_rate_{filename}.png")

    with h5py.File(output_path + f'{filename}.hdf5', "w") as hdf_file:
        hdf_file.create_dataset('V', data=Vhist)
        hdf_file.create_dataset('spike_rate', data=spike_rate)
        hdf_file.create_dataset('Weights', data=W)
        hdf_file.create_dataset('Centers', data=(C - 0.5*duration) )
        hdf_file.create_dataset('Sigmas', data=S)
        for name, value in params.items():
            hdf_file.attrs[name] = value


def multipal_run(output_path, W, C, S):
    # Параметры модели: частота тета-ритма, скорость животного, размера поля места (сигма), веса входов, их центры и сигмы.
    # run optimizeed model with different params
    dt = 0.1
    duration = 10000
    run_model_args = []
    for param_name in ['theta_freq', 'animal_velosity']:
         for param_var in globals()[param_name]:
             param = copy(default_param)
             param[param_name] = param_var
             filename = param_name + str(param_var)
             run_model_args.append((param, default_param, W, C, S, dt, duration, output_path, filename) )

    with Pool(processes=4) as p:
        p.map( run_model_with_parameters, run_model_args)

if __name__ == '__main__':
    precession_slope = [2.5, 3.5, 5, 6, 7]
    animal_velosity = [10, 15, 20, 25, 30]
    R_place_cell = [0.4, 0.5, 0.55]
    sigma_place_field = [2, 3, 4, 5]
    theta_freq = [4, 6, 8, 10, 12]


    default_param = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 4,
                     'theta_freq': 8, "use_x0": True}

    # optimize to the default params
    # main("test5", default_param)

    output_path = './output/tests/research/' # './output/default_optimization/'
    optimized_file = './output/tests/test_experiment_small_sigmas.hdf5' #  './output/default_optimization/default_experiment.hdf5'

    with h5py.File(optimized_file, "r") as hdf_file:
        W = hdf_file['Weights'][:]
        C = hdf_file['Centers'][:]
        S = hdf_file['Sigmas'][:]
    multipal_run(output_path, W, C, S)


    #####################################################################################
    # Experiment for research the role of each optimized parameters
    # datafile = "inputs_data.csv"
    # data = pd.read_csv(datafile, header=0, comment="#", index_col=0)
    #
    # # optimized_path = './output/default_optimization/'
    # optimized_path = './output/multipal_optimization/'
    #
    # # optimized_files = ['default_experiment.hdf5', ]
    # optimized_files = os.listdir(optimized_path)
    #
    # for optimized_file in optimized_files:
    #      optimized_file_name, optimized_file_ext = os.path.splitext(optimized_file)
    #      if optimized_file_ext != '.hdf5':
    #          continue
    #
    #      with h5py.File(optimized_path + optimized_file, "r") as hdf_file:
    #           W = hdf_file['Weights'][:]
    #           C = hdf_file['Centers'][:]
    #           S = hdf_file['Sigmas'][:]
    #
    #      output_path =  f'{optimized_path}{optimized_file_name}/research/'
    #
    #      try:
    #         os.makedirs(output_path)
    #      except:
    #         pass
    #      multipal_run(output_path, W, C, S)


         # for pop_idx, pop in enumerate(data.columns):
         #     print(pop)
         #
         #     for pop_w in np.arange(0, 1.1, 0.1):
         #         pop_w = np.around(pop_w, 1)
         #         output_path = f'{optimized_path}{optimized_file_name}/weights/{pop}/{pop_w}/'
         #         try:
         #             os.makedirs(output_path)
         #         except:
         #             pass
         #         Wz = np.copy(W)
         #         Wz[pop_idx] = pop_w
         #         multipal_run(output_path, Wz, C, S)

    #     for pop_w in np.arange(0, 1.1, 0.1):
    #         pop_w = np.around(pop_w, 1)
    #         output_path = f'{optimized_path}{optimized_file_name}/weights/all/{pop_w}'
    #         try:
    #             os.makedirs(output_path)
    #         except:
    #             pass
    #         Wz = np.copy(W)
    #         Wz[:2] = pop_w
    #         multipal_run(output_path, Wz, C, S)



    ##############################################################################
    ### Experiments for research parametres of spike rate function
    # for i in range(1, 10):
    #     param = copy(default_param)
    #     param['precession_slope'] = np.random.uniform(precession_slope[0], precession_slope[-1]) # precession_slope
    #     param['sigma_place_field'] = np.random.uniform(sigma_place_field[0], sigma_place_field[-1])
    #
    #     name = f'experiment_{i}'
    #     param["use_x0"] = False
    #     main(name, param)




