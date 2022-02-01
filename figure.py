import h5py
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phase_precession_v5 import get_teor_spike_rate, r2kappa
import scipy.signal as signal
from scipy.signal import hilbert, filtfilt
from scipy.optimize import minimize_scalar
from scipy.signal.windows import parzen
from scipy.stats import spearmanr, pearsonr
import os
import shutil

def fig2(name_file, param_local):
    '''
    A. График мембранного потенциала на теле нейрона. 
    B. Теоретическая и симулированная частота разрядов с наложенной синусоидой. 
    C. Фаза разрядов относительно тета-ритма в зависимости от положения животного в пространстве. 
    D. Проводимость каждого из входов.  
        Отдельный график для суммы возбуждающих проводимостей и тормозных проводимостей.
    '''

    duration = 3000
    dt = 0.1

    fig = plt.figure(figsize=(19,9))
    # ax1 = fig.add_subplot(1,1,1)
    ax1 = fig.add_subplot(5,2,1)
    ax2 = fig.add_subplot(5,2,2)
    ax3 = fig.add_subplot(5,1,2)
    ax4 = fig.add_subplot(5,2,5)
    ax5 = fig.add_subplot(5,2,6)
    fig2A(ax1, name_file, duration, dt)
    fig2B(ax3, name_file, duration, dt)
    fig2C(ax2, name_file, duration, dt)
    param = {'mode': 'inhibitory', 'num': 0}
    fig2D(ax5, name_file, duration, dt, param)
    param['mode'] = 'excitatory'
    param['num'] = 1
    fig2D(ax4, name_file, duration, dt, param)
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(5,4,13+i*4+j)
            param['mode'] = f'{i*4+j}'
            param['num'] = j
            fig2D(ax, name_file, duration, dt, param)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(f'output/fig2/{param_local["num"]}')
    plt.show()
    # fig.savefig(f"{name_file}.png")

def fig2A(ax, name_file, duration, dt):
    fs = 1000 / dt

    with h5py.File(f'{name_file}', 'r') as hdf_file:
        V = hdf_file['V'][:]

    sim_time = np.linspace(-0.5 * duration, 0.5 * duration, V.size)
    b, a = signal.butter(2, [4, 12], 'bandpass', fs = fs, output='ba')
    V_filt_theta = filtfilt(b, a, V)

    b, a = signal.butter(2, 2, 'low', fs = fs)
    # V_filt_slow = np.abs(hilbert(filtfilt(b, a, V)))
    V_filt_slow = filtfilt(b, a, V)

    ax.plot(sim_time, V, label="Raw potential")
    ax.plot(sim_time, V_filt_theta-np.max(V_filt_theta)-80, label="Theta comment")
    ax.plot(sim_time, V_filt_slow-np.max(V_filt_slow)-70, label="Slow comment")
    ax.set_title('A', loc='left')
    ax.set(xlabel='t, ms', ylabel='V, mV', xlim=[-0.5*duration, 0.5*duration])
    ax.legend(loc='upper right')
    ax.grid()

    return ax

def fig2B(ax, name_file, duration, dt):


    with h5py.File(f'{name_file}', 'r') as hdf_file:
        spike_rate = hdf_file['spike_rate'][:]
        precession_slope = hdf_file.attrs['precession_slope']
        theta_freq = hdf_file.attrs['theta_freq']
        R_place_cell = hdf_file.attrs['R_place_cell']
        animal_velosity = hdf_file.attrs['animal_velosity']
        sigma_place_field = hdf_file.attrs['sigma_place_field']
        teor_spike_rate = hdf_file['teor_spike_rate'][:]
        V = hdf_file['V'][:]
    sim_time = np.linspace(-0.5 * duration, 0.5 * duration, V.size)
    


    # precession_slope = animal_velosity * np.deg2rad(precession_slope)

    cos_ref = (np.cos(2*np.pi*theta_freq*0.001*sim_time)+1)/2
    index_teor, _ = signal.find_peaks(teor_spike_rate, height=0.1)
    index_exp, _ = signal.find_peaks(spike_rate, height=0.1)

    firing_idxs, _ = signal.find_peaks(V, height=-10)


    ax.plot(sim_time, teor_spike_rate,  linewidth=1, label='target spike rate')
    ax.plot(sim_time, spike_rate, linewidth=1, label='simulated spike rate')
    ax.plot(sim_time, cos_ref, linestyle = '--', linewidth=0.5)
    ax.scatter(sim_time[index_teor], cos_ref[index_teor])
    ax.scatter(sim_time[firing_idxs], cos_ref[firing_idxs])

    ax.legend(loc='upper right')
    ax.set_title('B', loc='left')
    ax.set(xlabel='t, ms', ylabel='$f/f_0$', xlim=[-0.5*duration, 0.5*duration])
    # ax.legend()
    ax.grid()

    return ax

def fig2C(ax, name_file, duration, dt):


    with h5py.File(f'{name_file}', 'r') as hdf_file:
        theta_freq = hdf_file.attrs['theta_freq']
        V = hdf_file['V'][:]
        animal_velosity = hdf_file.attrs['animal_velosity']


    sim_time = np.linspace(-0.5 * duration, 0.5 * duration, V.size)
    firing_idxs, _ = signal.find_peaks(V, height=-10)

    firing = sim_time[firing_idxs]
    animal_position = firing*animal_velosity*0.001
    phases_firing = 2*np.pi*theta_freq*firing*0.001

    sl = np.rad2deg(get_slope(phases_firing, animal_position))
    phases_firing = np.rad2deg(phases_firing%(2*np.pi))



    ax.scatter(animal_position, phases_firing, s=5, label=f'slope = {sl:0.1f}$^0/cm$')
    # ax.set_label('Label via method')
    ax.set_title('C', loc='left')
    position_start = -0.5*duration*0.001*animal_velosity
    ax.set(xlabel='animal position, cm', ylabel='$\Delta \\varphi, ^{\circ}$', xlim=[position_start, -position_start], ylim=[0, 360])
    ax.legend(loc='upper right')
    ax.grid()

    return ax

def fig2D(ax, name_file, duration, dt, param):


    data = pd.read_csv('inputs_data.csv', header=0, comment="#", index_col=0)

    sim_time = np.arange(0, duration, dt) - 0.5*duration

    with h5py.File(f'{name_file}', 'r') as hdf_file:
        W = hdf_file['Weights'][:]
        S = hdf_file['Sigmas'][:]
        C = hdf_file['Centers'][:]

    g_syn = np.zeros((sim_time.size, len(data.columns)), dtype=np.float64)
    with h5py.File('./output/conductances.hdf5', "r") as hdf_file:
        for inp_idx, input_name in enumerate(data.columns):
            g_syn[:, inp_idx] = hdf_file[input_name][:]

    g_syn_wcs = np.copy(g_syn)


    for idx in range(g_syn_wcs.shape[1]):
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ( (C[idx]-sim_time) / S[idx])**2 )
    
    # g = np.sum(g_syn_wcs, axes=1)

    if param['mode'] == 'all':
        g = np.sum(g_syn_wcs, axis=1)
        label = 'all'
    elif param['mode'] == 'excitatory':
        g = np.sum(g_syn_wcs[:, :2], axis=1)
        label = 'excitatory'
        ax.set_title('D', loc='left')
    elif param['mode'] == 'inhibitory':
        g = np.sum(g_syn_wcs[:, 2:], axis=1)
        label = 'inhibitory'
    else:
        g = g_syn_wcs[:, int(param['mode'])]
        label = data.columns[int(param['mode'])]
        if param['mode'] == '0':
            ax.set_title('E', loc='left')
    ax.plot(sim_time, g, label=label)
    # if label == 'cck':
    #     ax.plot(sim_time, g_syn_wcs[:, 3], label='PV')



    if param['num'] == 0:
        ax.set(xlabel='t, ms', ylabel='g, nS', xlim=[sim_time[0], sim_time[-1]])
    else:
        ax.set(xlabel='t, ms', xlim=[sim_time[0], sim_time[-1]])
    ax.set_ylim(0, 1.5 * np.max(g))
    ax.legend(loc='upper right')
    ax.grid()

    return ax

def fig3(directory):

    duration = 3000
    dt = 0.1
    Vreset = -80

    fig = plt.figure()
    # ax1 = fig.add_subplot(84)

    param = {'mode': 'sl', 'type': 1, 'name': 'C', 'num': 0}
    plot_param = {'x': '', 'title': ''}
    name = ['C', 'S', 'W', 'animal_velosity', 'theta_freq']
    x = {'C': 'Center, cm', 'S': 'Sigma, cm', 'W': 'Weight, 1', 'animal_velosity': 'v, cm/c', 'theta_freq': '$\omega_{\\theta}, Hz$'}

    ax = fig.add_subplot(4,2,1)
    param['type'] = 0
    param['name'] = 'animal_velosity'
    plot_param['x'] = x['animal_velosity']
    plot_param['title'] = 'A'
    fig3A(ax, directory, duration, dt, Vreset, param, plot_param)

    ax = fig.add_subplot(4,2,2)
    param['name'] = 'theta_freq'
    plot_param['x'] = x['theta_freq']
    plot_param['title'] = 'B'
    fig3A(ax, directory, duration, dt, Vreset, param, plot_param)

    param['type'] = 1
    voc = ['C', 'D', 'E']
    for i, tmp in enumerate(['W', 'S', 'C']):
        for j in range(8):
            ax = fig.add_subplot(4, 8, (i+1)*8+j+1)
            param['name'] = tmp
            param['num'] = j
            # print(param)
            plot_param['x'] = x[tmp]
            if j == 0: 
                plot_param['title'] = voc[i]
            else:
                plot_param['title'] = ''
            fig3A(ax, directory, duration, dt, Vreset, param, plot_param)

    
    # fig3A(ax1, directory, duration, dt, Vreset, param, plot_param)
    plt.subplots_adjust(wspace=0.45, hspace=0.5)
    plt.tight_layout()
    plt.show()
    
def get_phi_0(slope, phi_train, x_train):
    s = np.sum(np.cos(phi_train - 2 * np.pi * slope * x_train))
    s += 1j * np.sum(np.sin(phi_train - 2 * np.pi * slope * x_train))
    phi_0 = np.angle(s)
    return phi_0

def get_Dist(slope, phi_train, x_train):
    phi_0 = get_phi_0(slope, phi_train, x_train)
    D = 2 * (1 - np.mean(np.cos(phi_train - 2 * np.pi * slope * x_train - phi_0)))
    return D
    
def get_slope(phases_train, x_train):
    sl = np.linspace(-0.5, 0.5, 10000)
    D = []
    for s in sl:
         D.append(get_Dist(s, phases_train, x_train))

    slope = sl[np.argmin(D)]

    # res = minimize_scalar(get_Dist, x0=x0, args=(phases_train, x_train), bounds=[-0.5, 0.5], method='bounded')
    # slope = float(res.x)

    return slope

def correl(phases_train, x_train):
    r_xs = pearsonr(np.sin(phases_train), x_train)[0]
    r_xc = pearsonr(np.cos(phases_train), x_train)[0]
    r_cs = pearsonr(np.sin(phases_train), np.cos(phases_train))[0]
    # print(r_xs, r_xc, r_cs)
    cor = ((r_xc**2 + r_xs**2 - 2*r_xc*r_xs*r_cs)/(1 - r_cs**2))**0.5
    return cor

def get_slr(V, f, vel, duration, dt):
    sim_time = np.arange(0, duration, dt)

    peaks, _ = signal.find_peaks(V, height=(20, 30))

    t = sim_time[peaks]
    ph = 2*np.pi*f*t*0.001
    x = t*vel*0.001

    sl = get_slope(ph, x)
    r = correl(ph, x)
    # print(sl, r)

    return (sl, r)

def get_data(directory, duration, dt, Vreset, param):
    directory += f'/{param["name"]}'
    files = os.listdir(directory)
    files = [file for file in files if file[-4:] == 'hdf5']
    p = []
    r = []
    sl = []
    for name_file in files:
        if (param['type'] == 1 and f'{param["name"]}_{param["num"]}' in name_file) or param['type'] == 0:
            # print(name_file)
            with h5py.File(f'{directory}/{name_file}', 'r') as hdf_file:
                f = hdf_file.attrs['theta_freq']
                vel = hdf_file.attrs['animal_velosity']
                V = hdf_file['V'][:]
                # teor_spike_rate = hdf_file['teor_spike_rate'][:]
                if param['type'] == 0:
                    p.append(hdf_file.attrs[param['name']])
                elif param['type'] == 1:
                    s = name_file.split('_')
                    if s[0] == 'W':
                        name = 'Weights'
                    elif s[0] == 'S':
                        name = 'Sigmas'
                    elif s[0] == 'C':
                        name = 'Centers'
                    num = param['num']
                    p.append(hdf_file[name][num])

            sl_, r_ = get_slr(V, f, vel, duration, dt, Vreset)
            sl.append(np.rad2deg(sl_))
            r.append(r_)

    return p, r, sl
            
def make_folders(directory):
    param = ['W', 'S', 'C', 'animal_velosity', 'theta_freq']
    files = os.listdir(directory)
    # print(files)
    for p in param:
        if p not in files:
            os.mkdir(f'{directory}/{p}')
    for file in files:
        for p in param:
            if p in file:
                shutil.move(f'{directory}/{file}', f'{directory}/{p}')

def fig3A(ax, directory, duration, dt, Vreset, param, plot_param):

    p, r, sl = get_data(directory, duration, dt, Vreset, param)   

    if param['mode'] == 'r':
        ax.scatter(p, r, s=5)
        ylabel = 'r'
    if param['mode'] == 'sl':
        ax.scatter(p, sl, s=5)
        ylabel = '$slope, ^{\circ}/cm$'
    # ax.set_label('Label via method')
    ax.set_title(plot_param['title'], loc='left')
    if param['num'] == 0:
        ax.set(xlabel=plot_param['x'], ylabel=ylabel)
    else:
        ax.set(xlabel=plot_param['x'])
    # p_1, v_1 = np.polyfit(x[2:], ph[2:], deg=1, cov=True)
    # p_f1 = np.poly1d(p_1)
    # # # v_f1 = np.poly1d([v_1[0][0], v_1[1][1]])
    # print(p_f1)
    # ax.plot(x, list(p_f1(x)))
    # ax.legend(loc='upper left')
    ax.grid()

    return ax

def fig4(directory):

    # ax1 = fig.add_subplot(84)

    files = os.listdir(directory)
    files = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
    # print(files)

    param = {'mode': '', 'name': 'C', 'num': 0}
    plot_param = {'x': '', 'title': '', 'y': ''}
    name = ['animal_velosity', 'theta_freq', 'R_place_cell', 'precession_slope', 'sigma_place_field']
    x = {'animal_velosity': 'v, cm/c', 'theta_freq': '$\omega_{\\theta}, Hz$', 'R_place_cell': 'R, cm', 'sigma_place_field': '\sigma_{place field}, cm', 'precession_slope': 'sl_{teor}, ^{\circ}/cm'}
    y = {'C': 'Center, cm', 'S': 'Sigma, cm', 'W': 'Weight, 1'}

    # name = name[1]
    for tmp1 in name:
        fig = plt.figure(figsize=(19, 9))
        # print(tmp)
        param['mode'] = tmp1
        voc = ['A', 'B', 'C']
        # print(x[param['mode']])
        plot_param['x'] = x[param['mode']]
        for i, tmp in enumerate(['W', 'S', 'C']):
            param['name'] = tmp
            plot_param['y'] = y[tmp]
            for j in range(8):
                ax = fig.add_subplot(3, 8, (i)*8+j+1)
                param['num'] = j
                # print(param)
                if j == 0: 
                    plot_param['title'] = voc[i]
                else:
                    plot_param['title'] = ''
                fig4A(ax, directory, files, param, plot_param)

        plt.subplots_adjust(wspace=0.45, hspace=0.5)
        plt.tight_layout()
        plt.savefig(f'output/{tmp1}')
        # plt.show()
    

def fig4A(ax, directory, files, param, plot_param):

    r = []
    p = []

    for file in files:
        with h5py.File(f'{directory}/{file}', 'r') as hdf_file:
            p.append(hdf_file.attrs[param['mode']])
            if param['name'] == 'W':
                name = 'Weights'
            elif param['name'] == 'S':
                name = 'Sigmas'
            elif param['name'] == 'C':
                name = 'Centers'
            num = param['num']
            r.append(hdf_file[name][num])
      
    # ax.set_label('Label via method')
    if param['name'] == 'W':
        ax.set_ylim([0, 1])
    ax.set_title(plot_param['title'], loc='left')
    if param['num'] == 0:
        ax.set(xlabel=plot_param['x'], ylabel=plot_param['y'])
    else:
        ax.set(xlabel=plot_param['x'])
    ax.scatter(p, r)
    # p_1, v_1 = np.polyfit(x[2:], ph[2:], deg=1, cov=True)
    # p_f1 = np.poly1d(p_1)
    # # # v_f1 = np.poly1d([v_1[0][0], v_1[1][1]])
    # print(p_f1)
    # ax.plot(x, list(p_f1(x)))
    # ax.legend(loc='upper left')
    ax.grid()

    return ax    

def fig2_for_exp_4():
    directory = './output'

    files = os.listdir(directory)
    files = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
    i = 0
    param = {'num': i}
    files = files[:1]
    for file in files:
        param['num'] = i
        fig2(f'{directory}/{file}', param)
        i += 1
    
def main():
    # directory = 'phase_precession_results/output'
    # directory = 'output/research_default_optimization'
    # directory = 'output/multipal_optimization'

    fig2_for_exp_4()

    # directory = 'output'
    # files = os.listdir(directory)
    # # print(files)
    # images = [x for x in files if x[-5:] == '.hdf5']
    # print(*enumerate(images))
    # N = int(input('num: '))
    # name_file = f'{directory}/{images[N][:-5]}'
    # print(name_file)
    # s = input('key: ')
    # name_file = ''

    # fig4(directory)
    # make_folders(directory)
    # fig2(name_file)

    # fig4(directory)



if __name__ == '__main__':
    main()