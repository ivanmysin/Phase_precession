import h5py

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import pandas as pd
from phase_precession import get_teor_spike_rate, r2kappa
import scipy.signal as signal

def fig2(name_file):
    '''
    A. График мембранного потенциала на теле нейрона. 
    B. Теоретическая и симулированная частота разрядов с наложенной синусоидой. 
    C. Фаза разрядов относительно тета-ритма в зависимости от положения животного в пространстве. 
    D. Проводимость каждого из входов.  
        Отдельный график для суммы возбуждающих проводимостей и тормозных проводимостей.
    '''

    duration = 3000
    dt = 0.1

    # print(1)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))

    fig = plt.figure()

    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(312)
    ax4 = fig.add_subplot(325)
    ax5 = fig.add_subplot(326)
    fig2D(ax5, 'neg', name_file, duration, dt)
    fig2D(ax4, 'pos', name_file, duration, dt)
    fig2B(ax3, name_file, duration, dt)
    fig2C(ax2, name_file, duration, dt)
    fig2A(ax1, name_file, duration, dt)


    plt.tight_layout()
    plt.show()

def fig2A(ax, name_file, duration, dt):
    sim_time = np.arange(0, duration, dt)

    with h5py.File(f'{name_file}.hdf5', 'r') as hdf_file:
        V = hdf_file['V'][:]
        
    ax.plot(sim_time, V)
    ax.set_title('A', loc='left')
    ax.set(xlabel='t, ms', ylabel='V, mV', xlim=[0, duration])
    # ax.legend(loc='upper left')
    ax.grid()

    return ax

def fig2B(ax, name_file, duration, dt):
    sim_time = np.arange(0, duration, dt)

    with h5py.File(f'{name_file}.hdf5', 'r') as hdf_file:
        spike_rate = hdf_file['spike_rate'][:]
        V = hdf_file['V'][:]
        teor_spike_rate = hdf_file['teor_spike_rate'][:]
        precession_slope = hdf_file.attrs['precession_slope']
        theta_freq = hdf_file.attrs['theta_freq']
        R_place_cell = hdf_file.attrs['R_place_cell']
        animal_velosity = hdf_file.attrs['animal_velosity']
        sigma_place_field = hdf_file.attrs['sigma_place_field']
    
    place_field_center = 30

    precession_slope = animal_velosity * np.deg2rad(precession_slope)
    kappa_place_cell = r2kappa(R_place_cell)
    sigma_place_field = sigma_place_field / animal_velosity # recalculate to sec
    place_field_center = place_field_center / animal_velosity

    # teor_spike_rate = get_teor_spike_rate(sim_time, precession_slope, theta_freq, kappa_place_cell,  sigma=sigma_place_field, center=place_field_center)
    y = (np.cos(2*np.pi*theta_freq*0.001*sim_time)+1)/2
    index_teor, _ = signal.find_peaks(teor_spike_rate, height=0.1)
    index_exp, _ = signal.find_peaks(V, height=-10)  # signal.argrelmax(spike_rate)

    ax.plot(sim_time, teor_spike_rate,  linewidth=1, label='target spike rate')
    ax.plot(sim_time, spike_rate, linewidth=1, label='simulated spike rate')
    ax.plot(sim_time, y, linestyle = '--')
    ax.scatter(sim_time[index_teor], y[index_teor])
    ax.scatter(sim_time[index_exp], y[index_exp])

    ax.legend(loc='upper left')
    ax.set_title('B', loc='left')
    ax.set(xlabel='t, ms', xlim=[0, duration])
    # ax.legend()
    ax.grid()

    return ax

def fig2C(ax, name_file, duration, dt):
    sim_time = np.arange(0, duration, dt)
    Vreset = -80

    with h5py.File(f'{name_file}.hdf5', 'r') as hdf_file:
        f = hdf_file.attrs['theta_freq']
        V = hdf_file['V'][:]

    f *= 1
    T = 1/f

    t, _ = signal.find_peaks(V, height=-10)  # sim_time[V == Vreset]
    t = t * dt
    # print(t)
    ph = f*(t - t//T*T)*360
    # print()


    ax.scatter(t, ph, s=5)
    ax.set_label('Label via method')
    # ax.title(label='C', loc='left')
    ax.set_title('C', loc='left')
    ax.set(xlabel='t, ms', ylabel='$\Delta \\varphi, ^{\circ}$', xlim=[0, duration])
    # ax.legend(loc='upper left')
    ax.grid()

    return ax

def fig2D(ax, flag, name_file, duration, dt):
    data = pd.read_csv('inputs_data.csv', header=0, comment="#", index_col=0)

    sim_time = np.arange(0, duration, dt)

    with h5py.File(f'{name_file}.hdf5', 'r') as hdf_file:
        W = hdf_file['Weights'][:]
        S = hdf_file['Sigmas'][:]
        C = hdf_file['Centers'][:]

    g_syn = np.zeros((sim_time.size, len(data.columns)), dtype=np.float64)
    with h5py.File('./output/conductances.hdf5', "r") as hdf_file:
        for inp_idx, input_name in enumerate(data.columns):
            g_syn[:, inp_idx] = hdf_file[input_name][:]

    g_syn_wcs = np.copy(g_syn)

    # print(g_syn_wcs.shape)
    for idx in range(g_syn_wcs.shape[1]):
        # print(idx)
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ((C[idx] - sim_time) / S[idx]) ** 2)
    
    # g = np.sum(g_syn_wcs, axes=1)

    if flag == 'all':
        g = np.sum(g_syn_wcs, axis=1)
        label = 'all'
    elif flag == 'pos':
        g = np.sum(g_syn_wcs[:, :2], axis=1)
        label = 'pos'
        ax.set_title('D', loc='left')
    else:
        g = np.sum(g_syn_wcs[:, 2:], axis=1)
        label = 'neg'
    ax.plot(sim_time, g, label=label)
    ax.set(xlabel='t, ms', ylabel='g', xlim=[0, duration])
    ax.legend(loc='upper left')
    ax.grid()

    return ax



if __name__ == '__main__':
    name_file = 'output/default_experiment'
    fig2(name_file)