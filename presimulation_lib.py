import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
from default_params import get_inputs_names


########################################################################
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

def get_gsyns(data, theta_freq, sim_time):
    duration = sim_time[-1]
    g_syn = np.zeros((len(data.columns), sim_time.size), dtype=np.float64)
    Erev = np.zeros(len(data.columns), dtype=np.float64)

    for inp_idx, input_name in enumerate(data.columns):
        args = (data.loc["tau_rise"][input_name], data.loc["tau_decay"][input_name], data.loc["phi"][input_name],
            data.loc["kappa"][input_name], theta_freq)
        sol = solve_ivp(inegrate_g, t_span=[0, duration], t_eval=sim_time, y0=[0, 0], args=args, dense_output=True)
        g = sol.y[0]
        g *= 1.0 / np.max(g)
        g_syn[inp_idx, :] = g
        Erev[inp_idx] = data.loc["E"][input_name]

    return g_syn, Erev
#########################################################################
@jit(nopython=True)
def get_teor_spike_rate(t, slope, theta_freq, kappa, sigma=0.15, center=5):
    teor_spike_rate = np.exp(-0.5 * ((t - center)/sigma)**2 )
    precession = 0.001 * t * slope

    phi0 = -2 * np.pi * theta_freq * 0.001 * center - np.pi - precession[np.argmax(teor_spike_rate)]
    teor_spike_rate *= np.exp(kappa * np.cos(2*np.pi*theta_freq*t*0.001 + precession + phi0) )
    return teor_spike_rate
#########################################################################
@jit(nopython=True)
def r2kappa(R):
    if R < 0.53:
        return 2*R + R**3 + 5*R**5/6
    elif R >= 0.85:
        return 1/(3*R - 4*R**2 + R**3)
    else:
        return -0.4 + 1.39*R + 0.43/(1 - R)
#########################################################################
def get_soma_dend_idxes(data):
    inputs_names = get_inputs_names()
    soma_idxes = np.empty(0, dtype=np.int16)
    dend_idxes = np.empty(0, dtype=np.int16)

    for inp in inputs_names['soma']:
        soma_idxes = np.append(soma_idxes, data.columns.get_loc(inp))

    for inp in inputs_names['dend']:
        dend_idxes = np.append(dend_idxes, data.columns.get_loc(inp))
    return soma_idxes, dend_idxes