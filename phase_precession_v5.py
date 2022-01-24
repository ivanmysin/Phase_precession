import matplotlib
matplotlib.use('qt5agg')
import numpy as np
from numpy.random import randint
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import parzen
import time
from scipy.optimize import minimize, differential_evolution, shgo, dual_annealing
from scipy.integrate import solve_ivp
from numba import jit
import h5py
from tqdm.autonotebook import tqdm

@jit(nopython=True)
def inegrate_g(t, z, tau_rise, tau_decay, mu, kappa, freq):
    external_input = np.exp(kappa * np.cos(2*np.pi*t*freq*0.001 - mu) )
    g, dg = z
    tau_decay2 = tau_decay**2
    ddg  = (tau_rise*external_input - 2*tau_decay*dg-g)/tau_decay2
    out = np.empty(2, dtype=np.float64)
    out[0] = dg
    out[1] = ddg
    return out

def conv_g(g_syn, t, mask, W, S, C, R, mu):

    k = np.copy(R)
    for i in range(len(R)):
        k[i] = r2kappa(R[i])
    for idx in range(g_syn.shape[0]):
        g_syn[idx] *= W[idx] * np.exp(-0.5 * ((C[idx] - t) / S[idx]) ** 2)
        g_syn[idx] *= np.random.vonmises(mu[idx], k[idx], len(t))
        g_syn[idx] = np.convolve(g_syn[idx], mask)
    
    with h5py.File(f'conductance_{theta_freq}.hdf5', "w") as hdf_file:
            hdf_file.attrs["dt"] = dt
            hdf_file.attrs["duration"] = duration
            for inp_idx, input_name in enumerate(data.columns):
                args = (data.loc["tau_rise"][input_name], data.loc["tau_decay"][input_name], data.loc["phi"][input_name], data.loc["kappa"][input_name], theta_freq)
                sol = solve_ivp(inegrate_g, t_span=[0, duration], y0=[0, 0], max_step=dt, args=args, dense_output=True)
                g = sol.sol(sim_time)[0]
                g *= 0.1 / np.max(g)
                g_syn[:, inp_idx] = g
                Erev[inp_idx] = data.loc["E"][input_name]

                hdf_file.create_dataset(input_name, data=g)

    return g_syn


@jit(nopython=True)
def run_model(g_syn, sim_time, Erev, parzen_window, issaveV):
    El = -60
    VT = -50
    Vreset = -80
    gl = 0.1
    V = El
    C = 1
    dt = sim_time[1] - sim_time[0]
    spike_rate = np.zeros_like(sim_time)

    if issaveV:
        Vhist = np.zeros_like(sim_time)
    # print(Erev)
    for t_idx, t in enumerate(sim_time):
        Isyn = np.sum(g_syn[t_idx, :] * (Erev - V))
        # if t_idx < 50: print(Isyn)
        V = V + dt * (gl*(El - V) + Isyn) / C
        if V > VT:
            V = Vreset
            spike_rate[t_idx] = 1
            # print(t_idx)
        if issaveV:
            Vhist[t_idx] = V

    spike_rate = np.convolve(spike_rate, parzen_window)
    spike_rate = spike_rate[parzen_window.size//2:sim_time.size+parzen_window.size//2]
    if issaveV:
        return spike_rate, Vhist
    else:
        return spike_rate, np.empty(0)

@jit(nopython=True)
def get_teor_spike_rate(t, slope, theta_freq, kappa, sigma=0.15, center=5):
    t = 0.001 * t # ms to sec
    teor_spike_rate = np.exp(-0.5 * ((t - center)/sigma)**2 )
    precession = t * slope
    teor_spike_rate *= 0.5 * np.exp(kappa * np.cos(2*np.pi*theta_freq*t + precession) )
    return teor_spike_rate

@jit(nopython=True)
def r2kappa(R):
    if R < 0.53:
        return 2*R + R**3 + 5*R**5/6
    elif R >= 0.85:
        return 1/(3*R - 4*R**2 + R**3)
    else:
        return -0.4 + 1.39*R + 0.43/(1 - R)


@jit(nopython=True)
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

    # weitgh_balance = np.sum(W * np.sign(Erev + 2))
    # if weitgh_balance < 1:
    #      return np.exp(-10*weitgh_balance)
    # L = np.sum( (teor_spike_rate - spike_rate)**2 )
    L = np.mean( np.log( (teor_spike_rate+1)/(spike_rate+1) )**2 )
    return L

def run_with_param(num, param):
    print(f'start analysing for theta_freq={param["theta_freq"]}')
    ##################################################################
    ### Параметры для симуляции
    duration = 3000 # 5000 # 10000      # ms
    dt = 0.1              # ms
    theta_freq = param['theta_freq'] # 8 Hz
    precession_slope = param['precession_slope']  # deg/cm 5 
    animal_velosity = param['animal_velosity']  # cm/sec 20
    R_place_cell = param['R_place_cell']  # ray length 0.5
    place_field_center = 30 # 30 # cm
    sigma_place_field = param['sigma_place_field'] # cm 3
    ca3_center = 32 # 53.5   # 105
    ec3_center = 28 # 47.5   # 95
    datafile = "data_3.csv"
    # outfile = 'out.csv'
    conductance_file = "conductances"
    ###################################################################
    ### Делаем предвычисления
    sim_time = np.arange(0, duration, dt)
    precession_slope = animal_velosity * np.deg2rad(precession_slope)
    kappa_place_cell = r2kappa(R_place_cell)
    sigma_place_field = sigma_place_field / animal_velosity # recalculate to sec
    place_field_center = place_field_center / animal_velosity # recalculate to sec
    ca3_center = ca3_center / animal_velosity # recalculate to sec
    ec3_center = ec3_center / animal_velosity # recalculate to sec

    teor_spike_rate = get_teor_spike_rate(sim_time, precession_slope, theta_freq, kappa_place_cell,  sigma=sigma_place_field, center=place_field_center)
    
    data = pd.read_csv(datafile, header=0, comment="#", index_col=0)
    g_syn = np.zeros((sim_time.size, len(data.columns)), dtype=np.float64)
    Erev = np.zeros( len(data.columns), dtype=np.float64)
    parzen_window = parzen(151)
    data.loc["E"]  *= 1000
    if theta_freq not in [4, 6, 8, 10, 12]:
        data.loc["tau_rise"]  *= 1000
        data.loc["tau_decay"]  *= 1000
        data.loc["phi"]  = np.deg2rad(data.loc["phi"])
        data.loc["kappa"] = [ r2kappa(r) for r in data.loc["R"] ]
        ####################################################################
        # inegrate_g(t, z, tau_rise, tau_decay, mu, kappa, freq)
        print("start int synaptic coductances")
        with h5py.File(f'{conductance_file}_{theta_freq}.hdf5', "w") as hdf_file:
            hdf_file.attrs["dt"] = dt
            hdf_file.attrs["duration"] = duration
            for inp_idx, input_name in enumerate(data.columns):
                args = (data.loc["tau_rise"][input_name], data.loc["tau_decay"][input_name], data.loc["phi"][input_name], data.loc["kappa"][input_name], theta_freq)
                sol = solve_ivp(inegrate_g, t_span=[0, duration], y0=[0, 0], max_step=dt, args=args, dense_output=True)
                g = sol.sol(sim_time)[0]
                g *= 0.1 / np.max(g)
                g_syn[:, inp_idx] = g
                Erev[inp_idx] = data.loc["E"][input_name]

                hdf_file.create_dataset(input_name, data=g)
    else:
        print("read synaptic coductances")
        with h5py.File(f'{conductance_file}_{theta_freq}.hdf5', "r") as hdf_file:
            for inp_idx, input_name in enumerate(data.columns):
                g_syn[:, inp_idx] = hdf_file[input_name][:]
                Erev[inp_idx] = data.loc["E"][input_name]



    n_pops = len(data.columns)
    X = np.zeros(n_pops*3, dtype=np.float64)

    W = 0.1*np.ones(n_pops, dtype=np.float64)
    W[0] = 0.5
    W[1] = 0.5

    centers = np.zeros_like(W) + place_field_center*1000
    centers[0] = ca3_center*1000
    centers[1] = ec3_center*1000

    sigmas = np.zeros_like(centers) + sigma_place_field*1000
    X[0::3] = W
    X[1::3] = centers
    X[2::3] = sigmas

    print("start simulation")

    bounds = []
    for bnd_idx in range(X.size):
        if bnd_idx%3 == 0:
            bounds.append([0, 1])
        elif bnd_idx%3 == 1:
            bounds.append([0, duration])
        elif bnd_idx%3 == 2:
            bounds.append([0, 3000])

    # Изменяем границы для параметров для СА3
    bounds[0][0] = 0.1 # вес не менее 0,2
    bounds[1][0] = place_field_center*1000 # центр входа от СА3 не ранее центра в СА1

    # Изменяем границы для параметров для EC3
    bounds[3][0] = 0.1 # вес не менее 0,2
    bounds[4][1] = place_field_center*1000 # центр входа от EC3 ранее центра в СА1

    loss_args = (teor_spike_rate, g_syn, sim_time, Erev, parzen_window, False)

    timer = time.time()
    #mutation = (0.5, 1.9)
    sol = differential_evolution(loss, x0=X, popsize=1, atol=1e-6, recombination=0.5, \
                                mutation=1.7, args=loss_args, bounds=bounds,  maxiter=1, \
                                workers=1, updating='deferred', disp=True, \
                                strategy = 'best2bin')




    print("Time of optimization ", time.time() - timer, " sec")
    print("success ", sol.success)
    print("message ", sol.message)
    print("number of interation ", sol.nit)

    outdata = pd.DataFrame(columns = data.columns)
    outdata.loc["W"] = sol.x[0::3]
    outdata.loc["Center"] = sol.x[1::3]
    outdata.loc["Sigma"] = sol.x[2::3]
    # outdata.to_csv(f'{outfile}_{num}')

    g_syn_wcs = np.copy(g_syn)
    W = sol.x[0::3]
    C = sol.x[1::3]
    S = sol.x[2::3]

    # взвешивание и центрирование гауссианой
    for idx in range(n_pops):
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ((C[idx] - sim_time) / S[idx]) ** 2)
    spike_rate, Vhist = run_model(g_syn_wcs, sim_time, Erev, parzen_window, True)

    with h5py.File(f'{num}.hdf5', "w") as hdf_file:
        hdf_file.create_dataset('V', data=Vhist)
        hdf_file.create_dataset('spike_rate', data=spike_rate)
        hdf_file.create_dataset('W', data=W)
        hdf_file.create_dataset('C', data=C)
        hdf_file.create_dataset('S', data=S)
        for name, value in param.items():
            hdf_file.attrs[name] = value

def main_v1():
    precession_slope = [2.5, 3.5, 5, 6, 7]
    animal_velosity = [10, 15, 20, 25, 30]
    R_place_cell = [0.4, 0.5, 0.55]
    sigma_place_field = [2, 3, 4, 5]
    theta_freqs = [4, 6, 8, 10, 12]
    param = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 3, 'theta_freq': 8}

    lenth = [len(precession_slope), len(animal_velosity), \
            len(R_place_cell), len(sigma_place_field), \
            len(theta_freqs)]
    input_params = []
    for i in range(1):
        flag = False
        while not flag:
            n = np.ndarray.tolist(randint(lenth))
            if n not in input_params:
                flag = True
        n.insert(0, i)
        # print(n)
        input_params.append(n)
        # print(input_params)
        param['precession_slope'] = precession_slope[n[1]]
        param['animal_velosity'] = animal_velosity[n[2]]
        param['R_place_cell'] = R_place_cell[n[3]]
        param['sigma_place_field'] = sigma_place_field[n[4]]
        param['theta_freqs'] = theta_freqs[n[5]]
        name = f'experiment_{i}'
        run_with_param(name, param)
        
    # np.savetxt('exp_param', tmp)
    out = open('exp_param.csv', 'w')
    out.write(f'{"num"},{"pr_sl"},{"vel"},{"R"},{"sig"},{"theta"}\n')
    for voc in input_params:
        voc[1] = precession_slope[voc[1]]
        voc[2] = animal_velosity[voc[2]]
        voc[3] = R_place_cell[voc[3]]
        voc[4] = sigma_place_field[voc[4]]
        voc[5] = theta_freqs[voc[5]]
        out.write(','.join(map(str, voc))+'\n')
    out.close()

def run_for_3_fig(name_file, param, param_orig, W, C, S):
    print(f'start analysing for 3 figure')
    ##################################################################
    ### Параметры для симуляции
    duration = 3000 # ms
    dt = 0.1  # ms
    theta_freq = param['theta_freq'] # 8 Hz
    precession_slope = param_orig['precession_slope']  # deg/cm 5 
    animal_velosity = param['animal_velosity']  # cm/sec 20
    R_place_cell = param_orig['R_place_cell']  # ray length 0.5
    place_field_center = 30 # 30 # cm
    sigma_place_field = param_orig['sigma_place_field'] # cm 3
    
    datafile = 'data_3.csv'
    ###################################################################
    ### Делаем предвычисления
    sim_time = np.arange(0, duration, dt)
    precession_slope = param_orig['animal_velosity'] * np.deg2rad(precession_slope)
    kappa_place_cell = r2kappa(R_place_cell)
    sigma_place_field = sigma_place_field / animal_velosity # recalculate to sec
    place_field_center = place_field_center / animal_velosity # recalculate to sec
    
    teor_spike_rate = get_teor_spike_rate(sim_time, precession_slope, theta_freq, kappa_place_cell,  sigma=sigma_place_field, center=place_field_center)
    
    data = pd.read_csv(datafile, header=0, comment="#", index_col=0)
    g_syn = np.zeros((sim_time.size, len(data.columns)), dtype=np.float64)
    Erev = np.zeros( len(data.columns), dtype=np.float64)
    parzen_window = parzen(151)
    data.loc["E"]  *= 1000
    if theta_freq not in [4, 6, 8, 10, 12]:
        data.loc["tau_rise"]  *= 1000
        data.loc["tau_decay"]  *= 1000
        data.loc["phi"]  = np.deg2rad(data.loc["phi"])
        data.loc["kappa"] = [ r2kappa(r) for r in data.loc["R"] ]
        ####################################################################
        # inegrate_g(t, z, tau_rise, tau_decay, mu, kappa, freq)
        print("start int synaptic coductances")
        with h5py.File(f'conductances_{theta_freq}.hdf5', "w") as hdf_file:
            hdf_file.attrs["dt"] = dt
            hdf_file.attrs["duration"] = duration
            for inp_idx, input_name in enumerate(data.columns):
                args = (data.loc["tau_rise"][input_name], data.loc["tau_decay"][input_name], data.loc["phi"][input_name], data.loc["kappa"][input_name], theta_freq)
                sol = solve_ivp(inegrate_g, t_span=[0, duration], y0=[0, 0], max_step=dt, args=args, dense_output=True)
                g = sol.sol(sim_time)[0]
                g *= 0.1 / np.max(g)
                g_syn[:, inp_idx] = g
                Erev[inp_idx] = data.loc["E"][input_name]

                hdf_file.create_dataset(input_name, data=g)
    else:
        print("read synaptic coductances")
        with h5py.File(f'conductances_{theta_freq}.hdf5', "r") as hdf_file:
            for inp_idx, input_name in enumerate(data.columns):
                g_syn[:, inp_idx] = hdf_file[input_name][:]
                Erev[inp_idx] = data.loc["E"][input_name]

    # print(Erev)

    n_pops = len(data.columns)
    g_syn_wcs = np.copy(g_syn)

    # взвешивание и центрирование гауссианой
    for idx in range(n_pops):
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ((C[idx] - sim_time) / S[idx]) ** 2)

    spike_rate, Vhist = run_model(g_syn_wcs, sim_time, Erev, parzen_window, True)

    # fig, axes = plt.subplots(nrows=2, sharex=True)
    # axes[1].plot(sim_time, Vhist)
    # # axes[1].plot(sim_time, teor_spike_rate,  linewidth=1, label='target spike rate')
    # # axes[1].plot(sim_time, spike_rate, linewidth=1, label='simulated spike rate')
    # axes[0].plot(sim_time, np.sum(g_syn_wcs[:, :2], axis=1))
    # axes[0].plot(sim_time, np.sum(g_syn_wcs[:, 2:], axis=1))

    # plt.show()

    with h5py.File(f'output/exp3_{name_file}.hdf5', "w") as hdf_file:
        hdf_file.create_dataset('V', data=Vhist)
        hdf_file.create_dataset('spike_rate', data=spike_rate)
        hdf_file.create_dataset('teor_spike_rate', data=teor_spike_rate)
        hdf_file.create_dataset('W', data=W)
        hdf_file.create_dataset('C', data=C)
        hdf_file.create_dataset('S', data=S)
        for name, value in param.items():
            hdf_file.attrs[name] = value

def change_param(W, C, S, param):
    precession_slope = [2.5, 3.5, 5, 6, 7]
    animal_velosity = [10, 15, 20, 25, 30]
    sigma_place_field = [2, 3, 4, 5]
    theta_freqs = [4, 6, 8, 10, 12]

    lenth = [len(precession_slope), len(animal_velosity), \
            len(sigma_place_field), len(theta_freqs)]
    input_params = []
    for i in range(10):
        flag = False
        while not flag:
            n = np.ndarray.tolist(randint(lenth))
            if n not in input_params:
                flag = True
        n.insert(0, i)
        print(n)
        input_params.append(n)
        # print(input_params)
        param['precession_slope'] = precession_slope[n[1]]
        param['animal_velosity'] = animal_velosity[n[2]]
        param['sigma_place_field'] = sigma_place_field[n[3]]
        param['theta_freqs'] = theta_freqs[n[4]]
        name = f'exp3_{i}'
        # run_for_3_fig(name, param, W, C, S)
        
    # np.savetxt('exp_param', tmp)
    out = open('exp3.csv', 'w')
    out.write(f'{"num"},{"pr_sl"},{"vel"},{"sig"},{"theta"}\n')
    for voc in input_params:
        voc[1] = precession_slope[voc[1]]
        voc[2] = animal_velosity[voc[2]]
        voc[4] = sigma_place_field[voc[3]]
        voc[5] = theta_freqs[voc[4]]
        out.write(','.join(map(str, voc))+'\n')
    out.close()

def chan(x, par):
    return x*(1+par)  

def main_3f():
    param = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 3, 'theta_freq': 8}
    param_orig = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 3, 'theta_freq': 8}


    orig_param = pd.read_csv('orig_out.csv', header=0, comment="#", index_col=0)
    orig_param = orig_param.to_numpy()
    # print(orig_param)
    W = orig_param[0]
    C = orig_param[1]
    S = orig_param[2]
    # print(W, C, S)
    par = (np.arange(0, 21, 1)-10)*0.01
    # print(par)

    # print(chan(W[0], par[0]))\

    # num = 0
    # for w_, pw in zip(W, par):
    #     for s_, ps in zip(S, par):
    #         for c_, pc in zip(C, par):    
    #             num += 1
    #             w = chan(w_, pw)
    #             s = chan(s_, ps)
    #             c = chan(c_, pc)
                # run_for_3_fig(name, param, w, c, s)

    # for i in range(len(W)):
    #     tmp1 = W[:]
    #     for j in range(len(C)):
    #         for x in range(len(S)):

    #     for pv in par:
    #         tmp = voc[:]
    #         tmp[i] = chan(voc[i], pv)
    run_for_3_fig(f'orig', param, param_orig, W, C, S)
    

def int_g_theta():
    # надо закоментить строчки после итегрирования g в run_with_param
    theta_freqs = [4, 6, 8, 10, 12]
    param = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 3, 'theta_freq': 8}

    for theta in tqdm(theta_freqs):
        param['theta_freq'] = theta
        run_with_param(f'{theta}', param)

if __name__ == '__main__':
    main_3f()