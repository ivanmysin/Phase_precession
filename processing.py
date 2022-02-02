import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import scipy.signal as signal
import h5py
import shutil


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

def get_data(directory, duration, dt, param):
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

            sl_, r_ = get_slr(V, f, vel, duration, dt)
            sl.append(np.rad2deg(sl_))
            r.append(r_)

    return p, r, sl

def make_folders(directory):
    '''
    sort research_default_optimization into folders:
        W, S, C, animal_velosity, theta_freq
    for fig3
    '''
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

if __name__ == '__main__':
    pass