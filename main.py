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
from copy import copy
import os
from scipy import signal as sig
from multiprocessing import Pool
from phase_precession import r2kappa


class Pyramid:
    def __init__(self, inputs_data_file, dt, duration, params):
        self.dt = dt
        self.duration = duration

        self.sim_time = np.arange(0, duration, dt)
        self.sim_time = np.around(self.sim_time, 2)
        self.data = pd.read_csv(inputs_data_file, header=0, comment="#", index_col=0)
        self.params = params

        self.precompute_conductances()

        animal_velosity = params['animal_velosity']
        slope = animal_velosity *np.deg2rad( params['precession_slope'] )
        theta_freq = params['theta_freq']
        kappa = r2kappa(params['R_place_cell'])
        self.sigma = 1000 * params['sigma_place_field'] / animal_velosity
        center = 0.5*self.duration
        self.teor_spike_rate = self.get_teor_spike_rate(self.sim_time, slope, theta_freq, kappa, sigma=self.sigma, center=center)


    def precompute_conductances(self):

        self.data.loc["phi"] = np.deg2rad(self.data.loc["phi"])
        self.data.loc["kappa"] = [r2kappa(r) for r in self.data.loc["R"]]
        self.parzen_window = parzen(1001)

        theta_freq = self.params['theta_freq']
        self.g_syn = np.zeros((self.sim_time.size, len(self.data.columns)), dtype=np.float64)
        self.Erev = np.zeros(len(self.data.columns), dtype=np.float64)

        print("integrate synaptic coductances")
        for inp_idx, input_name in enumerate(self.data.columns):
            args = (self.data.loc["tau_rise"][input_name], self.data.loc["tau_decay"][input_name], self.data.loc["phi"][input_name],
                    self.data.loc["kappa"][input_name], theta_freq)
            sol = solve_ivp(self.inegrate_g, t_span=[0, duration], t_eval=self.sim_time, y0=[0, 0], args=args,
                                dense_output=True)
            g = sol.y[0]
            g *= 1.0 / np.max(g)
            self.g_syn[:, inp_idx] = g
            self.Erev[inp_idx] = self.data.loc["E"][input_name]

        self.soma_idxes = np.array([0, 2, 3, 7, 8]).astype(np.int16)
        self.dend_idxes = np.array([1, 4, 5, 6]).astype(np.int16)



    @staticmethod
    @jit(nopython=True)
    def inegrate_g(t, z, tau_rise, tau_decay, mu, kappa, freq):
        external_input = np.exp(kappa * np.cos(2 * np.pi * t * freq * 0.001 - mu))
        g, dg = z
        tau_decay2 = tau_decay ** 2
        ddg = (tau_rise * external_input - 2 * tau_decay * dg - g) / tau_decay2
        out = np.empty(2, dtype=np.float64)
        out[0] = dg
        out[1] = ddg
        return out

    def run_model(self, X=None):
        Erev = self.Erev + 60.0

        if X is None:
            g_syn = self.g_syn.T
        else:
            g_syn = np.copy(self.g_syn.T)
            W = X[0::3]
            C = X[1::3]
            S = X[2::3]
            # Weightings and centering
            for idx in range(W.size):
                g_syn[idx, :] *= W[idx] * np.exp(-0.5 * ((C[idx] - self.sim_time) / S[idx])**2)

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

            "input_conduntance": g_syn[self.soma_idxes, :],  # np.empty((0, 0), dtype=np.float32),
            "conduntances_Erev": Erev[self.soma_idxes],  # np.empty(0, dtype=np.float32),
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

            "input_conduntance": g_syn[self.dend_idxes, :],
            "conduntances_Erev": Erev[self.dend_idxes],

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


        pyramidal = lib.ComplexNeuron(neuron["compartments"], neuron["connections"])
        pyramidal.integrate(self.dt, self.sim_time[-1])
        Vsoma = pyramidal.getCompartmentByName('soma').getVhist()
        # Vdend = pyramidal.getCompartmentByName('dendrite').getVhist()
        self.spike_rate = np.zeros_like(Vsoma)

        peaks = find_peaks(Vsoma, height=30)[0]
        self.spike_rate[peaks] += 1
        self.spike_rate = np.convolve(self.spike_rate, self.parzen_window, mode='same')

        self.Vsoma = Vsoma - 60

        return self.spike_rate, self.Vsoma

    @staticmethod
    @jit(nopython=True)
    def get_teor_spike_rate(t, slope, theta_freq, kappa, sigma=0.15, center=5):
        # t = 0.001 * t # ms to sec
        teor_spike_rate = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        precession = 0.001 * t * slope

        phi0 = -2 * np.pi * theta_freq * 0.001 * center - np.pi - precession[np.argmax(teor_spike_rate)]
        teor_spike_rate *= np.exp(kappa * np.cos(2 * np.pi * theta_freq * t * 0.001 + precession + phi0))  # 0.5 *
        return teor_spike_rate

    def loss(self, X):
        # Xp = np.copy(X)
        # # W = Xp[0::3]
        # centers = Xp[1::3]
        # sigmas = Xp[2::3]
        #
        # centers[:] = centers * 1000 / self.params['animal_velosity']
        # sigmas[:] = centers * 1000 / self.params['animal_velosity']

        spike_rate, _ = self.run_model(X)
        L = np.mean( np.log( (self.teor_spike_rate+1)/(spike_rate+1) )**2 )
        return L

#############################################################
def sumloss(X, pyramidals):
    l = 0
    for pyr in pyramidals:
        l += pyr.loss(X)
    l /= len(pyramidals)
    return l



def minimization(pyramidals):
    pyr = pyramidals[0]

    n_pops = len(pyr.data.columns)
    place_field_center = 0.5*pyr.duration
    W = 0.1 * np.ones(n_pops, dtype=np.float64)
    W[0] = 0.5
    W[1] = 0.5

    centers = np.zeros_like(W) + place_field_center
    centers[0] += 1
    centers[1] -= 1

    sigmas = np.zeros_like(centers) + pyr.sigma

    if pyr.params["use_x0"]:
        X = np.zeros(n_pops * 3, dtype=np.float64)
        X[0::3] = W
        X[1::3] = centers
        X[2::3] = sigmas
    else:
        X = None

    print("start simulation")

    #max_center = 0.001 * duration * 10
    bounds = []
    for bnd_idx in range(n_pops * 3):
        if bnd_idx % 3 == 0:
            bounds.append([0, 1])
        elif bnd_idx % 3 == 1:
            bounds.append([0, duration])
        elif bnd_idx % 3 == 2:
            bounds.append([150, 50000])

    # Изменяем границы для параметров для СА3
    bounds[0][0] = 0.01  # вес не менее 0,2
    bounds[1][0] = place_field_center  # центр входа от СА3 не ранее центра в СА1

    # Изменяем границы для параметров для EC3
    bounds[3][0] = 0.01  # вес не менее 0,2
    bounds[4][1] = place_field_center  # центр входа от EC3 ранее центра в СА1

    timer = time.time()
    sol = differential_evolution(sumloss, x0=X, popsize=24, atol=1e-3, recombination=1.7, \
                                 mutation=1.2, bounds=bounds, maxiter=15, args=[pyramidals, ],\
                                 workers=-1, updating='deferred', disp=True, \
                                 strategy='best2bin')

    X = sol.x

    print("Time of optimization ", time.time() - timer, " sec")
    print("success ", sol.success)
    print("message ", sol.message)
    print("number of interation ", sol.nit)

    # spike_rate, V = pyr.run_model(X)

    plt.plot(pyr.sim_time, pyr.Vsoma)
    plt.show()
    return X

##############################################################
if __name__ == '__main__':
    animal_velosities = [20, ] # [10, 20, 30] # [10, 15, 20, 25, 30]
    theta_freqs = [4, 6, 8, 10]

    inputs_data_file = 'inputs_data.csv'
    output_path = './output/tests/'
    filename = 'summ_loss'
    dt = 0.1
    duration = 3000
    default_param = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 4,
                     'theta_freq': 8, "use_x0": False}

    timer = time.time()


    pyramidals = []
    for an_vel in animal_velosities:
        for theta_freq in theta_freqs:
            params = copy(default_param)
            params['animal_velosity'] = an_vel
            params['theta_freq'] = theta_freq
            pyr = Pyramid(inputs_data_file, dt, duration, params)
            pyramidals.append(pyr)

    X = minimization(pyramidals)
    print(X)

    W = X[0::3]
    C = X[1::3]
    S = X[2::3]

    for pyr_idx, pyr in enumerate(pyramidals):
        with h5py.File(output_path + f'{pyr_idx}.hdf5', "w") as hdf_file:
            hdf_file.create_dataset('V', data=pyr.Vsoma)
            hdf_file.create_dataset('spike_rate', data=pyr.spike_rate)
            hdf_file.create_dataset('Weights', data=W)
            hdf_file.create_dataset('Centers', data=(C - 0.5*duration) )
            hdf_file.create_dataset('Sigmas', data=S)
            for name, value in pyr.params.items():
                hdf_file.attrs[name] = value

