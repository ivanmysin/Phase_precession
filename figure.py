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
import os
import shutil
from processing import get_data, get_slr, make_folders
from scipy.optimize import minimize
from presimulation_lib import get_teor_spike_rate, r2kappa

neuron_colors = {
    "pyr" : '#ff4500', #orangered
    "pv": '#00008b', # darkblue
    "olm": '#6495ed', # cornflower   
    "cck": '#00ff00', # lime
    "ivy": '#006400', # darkgreen
    "ngf": '#ffd700', # gold
    "bis": '#00ffff', # aqua
    "aac": '#ffdab9', # peachpuff
    "ca3": '#ff00ff', # fuchsia
    "ec3": '#b03060', #maroon3
}

neuron_colors_voc = [
    '#ff00ff', # fuchsia ca3
    '#b03060', #maroon3 ec3
    '#00ff00', # lime cck
    '#00008b', # darkblue pv
    '#ffd700', # gold ngf
    '#006400', # darkgreen ivy
    '#6495ed', # cornflower olm
    '#ffdab9', # peachpuff aac
    '#00ffff', # aqua bis
]

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
    fig2C(ax2, name_file, {'all_folder': param_local['all_folder'], 'title': None, 'color': None, 'i': None}, duration, dt)
    param = {'mode': 'inhibitory', 'num': 0}
    fig2D(ax5, name_file, duration, dt, param)
    param['mode'] = 'excitatory'
    param['num'] = 0
    fig2D(ax4, name_file, duration, dt, param)
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(5,4,13+i*4+j)
            param['mode'] = f'{i*4+j}'
            param['num'] = j
            fig2D(ax, name_file, duration, dt, param)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if not os.path.exists('output/fig2'):
        os.makedirs('output/fig2') 
    plt.savefig(f'output/fig2/{param_local["num"]}')
    if param_local['flag']:
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

def fig2B(ax, name_file, duration=3000, dt=0.1):


    with h5py.File(f'{name_file}', 'r') as hdf_file:
        spike_rate = hdf_file['spike_rate'][:]

        if len(set(spike_rate)) == 2:
            parzen_window = parzen(1001)
            spike_rate = np.convolve(spike_rate, parzen_window, mode='same')

        theta_freq = hdf_file.attrs['theta_freq']
        teor_spike_rate = hdf_file['teor_spike_rate'][:]
        V = hdf_file['V'][:]

        if len(teor_spike_rate) == 0:
            sim_time = np.linspace(-0.5 * duration, 0.5 * duration, V.size)
            sl = hdf_file.attrs['precession_slope']
            R = hdf_file.attrs['R_place_cell']
            teor_spike_rate = get_teor_spike_rate(sim_time, sl, theta_freq, r2kappa(R))

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
    # ax.scatter(sim_time[index_exp], cos_ref[index_exp])

    ax.legend(loc='upper right')
    ax.set_title('B', loc='left')
    ax.set(xlabel='t, ms', ylabel='$f/f_0$', xlim=[-0.5*duration, 0.5*duration])
    # ax.legend()
    ax.grid()

    return ax


def fig2C(ax, name_file, local_param, duration=3000, dt=0.1, flag=True):
    animal_position = []
    phases_firing = []
    
    # print(local_param['all_folder'])

    if local_param['all_folder'] == 'None':
        files = [name_file]
    else:
        files = os.listdir(local_param['all_folder'])
        files = [f'{local_param["all_folder"]}/{file}' for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
        # if 'ec3' in local_param['all_folder']: print(files)
    # print(files)
    for file in files:
        with h5py.File(f'{file}', 'r') as hdf_file:
            theta_freq = hdf_file.attrs['theta_freq']
            V = hdf_file['V'][:]
            animal_velosity = hdf_file.attrs['animal_velosity']
            # print(type(V))

        firing_idxs, _ = signal.find_peaks(V, height=-10)
        sim_time = np.linspace(-0.5 * duration, 0.5 * duration, V.size)

        firing = sim_time[firing_idxs]
        animal_position_ = firing*animal_velosity*0.001
        phases_firing_ = 2*np.pi*theta_freq*firing*0.001 % (2*np.pi)
        animal_position.extend(animal_position_)
        phases_firing.extend(phases_firing_)
    
    # if 'ngf' in local_param['all_folder']: print(animal_position)

    # print(phases_firing, animal_position)

    if len(phases_firing) < 5: print(len(phases_firing))    
    if len(phases_firing) < 2: return ax, (0, 0, 0)

    phases_firing, animal_position = np.array(phases_firing), np.array(animal_position)
    sl, r, phi_0 = get_slr(phases_firing, animal_position)

    phases_firing = np.rad2deg(phases_firing%(2*np.pi))
    p_1, _ = np.polyfit(animal_position, phases_firing, deg=1, cov=True)
    # print(np.rad2deg(phi_0))

    p = np.poly1d([-np.rad2deg(sl), np.rad2deg(phi_0)+180])
    # print(p_1[1])
    # p = np.poly1d([-np.rad2deg(sl), p_1[1]])

    # solution = minimize(ob, 180, method='SLSQP')

    # if 'ngf' in local_param['all_folder']: print(sl, r)

    if flag:
        if local_param['color'] != None:
            # if 'ngf' in local_param['all_folder']: print(sl, r, local_param['title'])
            ax.scatter(animal_position, phases_firing, s=5, label=f'slope = {np.rad2deg(sl):0.1f}'+'$^{\circ}/cm$;'+f' \n corr = {r:0.5f}', color=local_param['color'])
            # ax.set_label('Label via method')
            ax.plot(animal_position[phases_firing < 200], list(p(animal_position[phases_firing < 200])), color=local_param['color'])
        else:
            ax.scatter(animal_position, phases_firing, s=5, label=f'slope = {np.rad2deg(sl):0.1f}'+'$^{\circ}/cm$;'+f' \n corr = {r:0.5f}')
            # ax.set_label('Label via method')
            ax.plot(animal_position[phases_firing < 200], list(p(animal_position[phases_firing < 200])))
        
        if local_param['title'] == None:
            ax.set_title('C', loc='left')
        else:
            ax.set_title(local_param['title'])
        position_start = -0.5*duration*0.001*animal_velosity

        if local_param['i'] != None:
            if local_param['i'] > 5:
                ax.set_xlabel('animal position, cm')
            if local_param['i']%3 == 0:
                ax.set_ylabel('$\Delta \\varphi, ^{\circ}$')
        ax.set(xlim=[-15, 15], ylim=[0, 360])
        ax.legend(loc='upper right')
        ax.grid()

    return ax, (sl, r, phi_0)

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
            if input_name != 'bis':
                # if input_name == 'ngf': print(hdf_file[input_name][:])
                g_syn[:, inp_idx] = hdf_file[input_name][:]


    g_syn_wcs = np.copy(g_syn)


    for idx in range(g_syn_wcs.shape[1]):
        g_syn_wcs[:, idx] *= W[idx] * np.exp(-0.5 * ( (C[idx]-sim_time) / S[idx])**2 )
    
    # g = np.sum(g_syn_wcs, axes=1)

    color = 'tab:blue'
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
        # ax.set_color(neuron_colors[label])
        # print(neuron_colors[label])
        # ax.tick_params(color=neuron_colors[label])
        color = neuron_colors[label]
    # if label == 'ngf': print(g)
    ax.plot(sim_time, g, label=label, color=color)
    # if label == 'cck':
    #     ax.plot(sim_time, g_syn_wcs[:, 3], label='PV')



    # if param['mode'] == 'inhibitory':
    #     ax.set(xlabel='t, ms', xlim=[sim_time[0], sim_time[-1]])
    # elif param['mode'] == 'excitatory':
    #     ax.set(xlabel='t, ms', ylabel='g, nS', xlim=[sim_time[0], sim_time[-1]])
    if param['num'] == 0:
        ax.set(xlabel='t, ms', ylabel='g, nS', xlim=[sim_time[0], sim_time[-1]])
    else:
        ax.set(xlabel='t, ms', xlim=[sim_time[0], sim_time[-1]])
    ax.set_ylim(0, 1.5 * np.max(g))
    ax.legend(loc='upper right')
    ax.grid()

    return ax

def fig3(directory, param_local):

    duration = 3000
    dt = 0.1

    fig = plt.figure(figsize=(19,9))
    # ax1 = fig.add_subplot(84)

    param = {'mode': param_local['mode'], 'type': 1, 'name': 'C', 'num': 0}
    plot_param = {'x': '', 'title': ''}
    name = ['C', 'S', 'W', 'animal_velosity', 'theta_freq']
    x = {'C': 'Center, cm', 'S': 'Sigma, cm', 'W': 'Weight, 1', 'animal_velosity': 'v, cm/c', 'theta_freq': '$\omega_{\\theta}, Hz$'}

    ax = fig.add_subplot(4,2,1)
    param['type'] = 0
    param['name'] = 'animal_velosity'
    plot_param['x'] = x['animal_velosity']
    plot_param['title'] = 'A'
    fig3A(ax, directory, duration, dt, param, plot_param)

    ax = fig.add_subplot(4,2,2)
    param['name'] = 'theta_freq'
    plot_param['x'] = x['theta_freq']
    plot_param['title'] = 'B'
    fig3A(ax, directory, duration, dt, param, plot_param)

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
            fig3A(ax, directory, duration, dt, param, plot_param)

    
    plt.subplots_adjust(wspace=0.45, hspace=0.5)
    plt.tight_layout()
    if param_local['flag']:
        plt.show()
    if not os.path.exists('output/fig3'):
        os.makedirs('output/fig3')
    plt.savefig(f'output/fig3/{param["mode"]}') 

def fig3A(ax, directory, duration, dt, param, plot_param):

    p, r, sl = get_data(directory, duration, dt, param)   

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
    x = {'animal_velosity': 'v, cm/c', 'theta_freq': '$\omega_{\\theta}, Hz$', 'R_place_cell': 'R, cm', 'sigma_place_field': '$\sigma_{place field}, cm$', 'precession_slope': '$sl_{teor}, ^{\circ}/cm$'}
    y = {'C': 'Center, cm', 'S': 'Sigma, cm', 'W': 'Weight, 1'}

    # name = name[1]
    if not os.path.exists('output/fig4'):
        os.makedirs('output/fig4') 

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
        plt.savefig(f'output/fig4/{tmp1}')
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

def fig2_for_exp_4(directory='./output/multipal_optimization', name=''):
    '''
    directory = multipal_optimization
    creates fig2 for all experiments 4 data
    '''
    # directory = './output/multipal_optimization'


    files = os.listdir(directory)
    files = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
    # print(files)
    i = 0
    param = {'num': f'{name}_{i}', 'flag': False, 'all_folder': 'None'}
    # files = files[:1]
    for file in files:
        print(file)
        param['num'] = f'{name}_{file[:-4]}'
        fig2(f'{directory}/{file}', param)
        i += 1

def new_fig3():
    directory = 'output/output/multipal_optimization'
    files = os.listdir(directory)
    files = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
    w = np.empty((0,9))
    s = np.empty((0,9))
    c = np.empty((0,9))
    for file in files:
        with h5py.File(f'{directory}/{file}', 'r') as hdf_file:
            # print(hdf_file['Weights'][:], w)
            v = hdf_file.attrs['animal_velosity']
            w = np.append(w, [hdf_file['Weights'][:]], axis=0)
            c = np.append(c, [hdf_file['Centers'][:]/1000*v], axis=0)
            s = np.append(s, [hdf_file['Sigmas'][:]/1000*v], axis=0)
            # np.append(hdf_file['Sigmas'][:], s, axis=0)
            # np.append(hdf_file['Centers'][:], c, axis=0)
            # print(w)

    # print(w)

    fig = plt.figure(figsize=(19,9))
    name = ['Weigths', 'Centers, cm', 'Sigmas, cm']
    voc = [w, c, s]
    title = ['A', 'B', 'C']
    inputs = ['ca3', 'ec3', 'cck', 'pv', 'ngf', 'ivy', 'olm', 'aac', 'bis']
    for i in range(3):
        ax = fig.add_subplot(3, 1, i + 1)
        # for j in range(9):
        #     ax = fig.add_subplot(3, 9, i*9 + j + 1)
        #     print(voc[i][:, j].shape)
        #     ax.boxplot(voc[i][:, j])
            # , labels=f'{name[i]} - {inputs[j]}')
        ax.set_title(title[i], loc='left')
        bp = ax.boxplot(voc[i], labels=inputs, patch_artist=True)
        for patch, color in zip(bp['boxes'], neuron_colors_voc):
            patch.set_facecolor(color)

        ax.yaxis.grid(True)
        if i == 2: ax.set_xlabel('Inputs')
        ax.set_ylabel(name[i])


    plt.subplots_adjust(wspace=0.45, hspace=0.5)
    plt.tight_layout()

    if True:
        plt.show()
    if not os.path.exists('output/fig3_new'):
        os.makedirs('output/fig3_new')

    plt.savefig(f'output/fig3_new/fig3.png') 


def new_fig4A():
    directory = 'output/output/default_optimization/weights'
    # folders = os.listdir(directory)
    # files = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']

    # print(folders)

    # fig2_for_exp_4(directory=f'{directory}/ec3/0.0/research', name='ec3')
    # name_file = f'{directory}/ec3/0.0/research/0.hdf5'
    # param_local = {'num': 0, 'flag': True, 'all_folder': f'{directory}/ngf/0.0/research'}
    # param_local['flag'] = True
    # fig2(name_file, param_local)
    # exit()

    fig = plt.figure(figsize=(19,9))
    inputs = ['ca3', 'ec3', 'cck', 'pv', 'ngf', 'ivy', 'olm', 'aac', 'bis']
    for i, folder in enumerate(inputs):
        s = f'{directory}/{folder}/0.0/research'
        files = os.listdir(s)
        file = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5'][0]
        ax = fig.add_subplot(3, 3, i+1)

        # if folder == 'ngf': print(file)

        fig2C(ax, f'{s}/{file}', {'all_folder': s, 'title': inputs[i], 'color': neuron_colors_voc[i], 'i': i})

        # ax.set_title(inputs[i])

    fig.suptitle('A', x=0.05)
    plt.subplots_adjust(wspace=0.45, hspace=0.5)
    plt.tight_layout()
    
    if True:
        plt.show()
    if not os.path.exists('output/fig4_new'):
        os.makedirs('output/fig4_new')

    plt.savefig(f'output/fig4_new/fig4A.png') 


def new_fig4B():
    directory = 'output/output/default_optimization/weights'

    fig = plt.figure(figsize=(19,9))
    inputs = ['ca3', 'ec3', 'cck', 'pv', 'ngf', 'ivy', 'olm', 'aac', 'bis']
    for i, folder in enumerate(inputs):
        w = []
        r = []

        folds = os.listdir(f'{directory}/{folder}')
        ax = fig.add_subplot(3, 3, i+1)
        for fold in folds:
            s = f'{directory}/{folder}/{fold}/research'
            files = os.listdir(s)
            file = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5'][0]
            w.append(float(fold))
            # r.append(fig2C(ax, f'{s}/{file}', {'all_folder': s, 'title': inputs[i], 'color': neuron_colors_voc[i]}, dt=1000))
            print(f'{s}/{file}')
            _, tmp = fig2C(ax, f'{s}/{file}', {'all_folder': s, 'title': inputs[i], 'color': neuron_colors_voc[i], 'i': i}, flag=False)
            r.append(tmp[1])


        # s = f'{directory}/{folder}/0.0/research'
        # files = os.listdir(s)
        # file = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
        # ax = fig.add_subplot(3, 3, i+1)

        # fig2C(ax, f'{s}/{file}', {'all_folder': s, 'title': inputs[i], 'color': neuron_colors_voc[i]})

        ax.set_title(inputs[i])

        if i > 5:
            ax.set_xlabel('Weigths')
        if i%3 == 0:
            ax.set_ylabel('Corr')
        ax.scatter(w, r, color=neuron_colors[folder])

    fig.suptitle('B', x=0.05)
    plt.subplots_adjust(wspace=0.45, hspace=0.5)
    plt.tight_layout()
    
    if True:
        plt.show()
    if not os.path.exists('output/fig4_new'):
        os.makedirs('output/fig4_new')

    plt.savefig(f'output/fig4_new/fig4B.png') 
    

def new_fig5():
    directory = 'output/output/small_sigma_multipal_optimization'
    files = os.listdir(directory)
    files = [file for file in files if (file[-4:] == 'hdf5') and file != 'conductances.hdf5']
    w = np.empty((0,9))
    c = np.empty((0,9))
    for file in files:
        with h5py.File(f'{directory}/{file}', 'r') as hdf_file:
            v = hdf_file.attrs['animal_velosity']
            w = np.append(w, [hdf_file['Weights'][:]], axis=0)
            c = np.append(c, [hdf_file['Centers'][:]/1000*v], axis=0)

    fig = plt.figure(figsize=(19,9))
    inputs = ['ca3', 'ec3', 'cck', 'pv', 'ngf', 'ivy', 'olm', 'aac', 'bis']
    for i, name in enumerate(inputs):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.scatter(c[:, i], w[:, i], color=neuron_colors[name])
        ax.set_title(name)
        if i > 5:
            ax.set_xlabel('Centers, cm')
        if i%3 == 0:
            ax.set_ylabel('Weigths')
        ax.set(xlim=[-25,25], ylim=[0,1])
        



    fig.suptitle('B', x=0.05)
    plt.subplots_adjust(wspace=0.45, hspace=0.5)
    plt.tight_layout()

    if True:
        plt.show()
    if not os.path.exists('output/fig5_new'):
        os.makedirs('output/fig5_new')

    plt.savefig(f'output/fig5_new/fig5.png') 



    
def main():
    ################
    # create fig2 for experement
    # create a folder 'output/fig2' if it doesn't exist
    # name_file = 'path/name_file'
    # param_local['all_folder'] : 'directory' - use all files
    #                             'None' - unused
    # param_local['flag'] : show figure
    # 
    # name_file = 'output/default_experiment.hdf5'
    # param_local = {'num': 0, 'flag': True, 'all_folder': 'None'}
    # param_local['flag'] = True
    # fig2(name_file, param_local)

    ################
    # create fig3 for experement 3:
    # 2 mode (param_local['mode']) : 'sl' - precession slope
    #         'r' - correlation (circular-linear)
    # param_local['flag'] : show figure
    # create a folder 'output/fig3' if it doesn't exist
    # path = output/research_default_optimization
    # 
    # param_local = {'mode': '', 'flag': True}
    # param_local['flag'] = False
    # param_local['mode'] = 'r'
    # directory = 'output/research_default_optimization'
    # fig3(directory, param_local)

    ################
    # sorts files into folders for experement 3
    # (output files are in the same folder)
    # folders:
    #       W, S, C, animal_velosity, theta_freq
    # path = output/research_default_optimization
    # 
    # directory = 'output/research_default_optimization'
    # make_folders(directory)

    ################
    # create fig4 for experement 4
    # create a folder 'output/fig4' if it doesn't exist
    # path = output/multipal_optimization
    # 
    # directory = 'output/multipal_optimization'
    # fig4(directory)

    ################
    # create fig2 for all experement 4
    # create a folder 'output/fig2' if it doesn't exist
    # name figure = {experiment number}.png
    # path = output/multipal_optimization
    # 
    # fig2_for_exp_4()

    ################
    # new_fig3()

    ################
    new_fig4A()

    ################
    # new_fig4B()

    ################
    # new_fig5()

if __name__ == '__main__':
    main()