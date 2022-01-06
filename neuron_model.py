import numpy as np
import matplotlib.pyplot as plt
import lib
import time

soma_params = {
    "V0": -5.0,
    "C": 3.0,
    "Iextmean": 1.5,
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

    "input_conduntance" : np.empty( (0, 0), dtype=np.float32),
    "conduntances_Erev" : np.empty(0, dtype=np.float32),
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

    "input_conduntance": np.empty( (0, 0), dtype=np.float32), # np.zeros( (1, 10000), dtype=np.float32) + 0.05, #
    "conduntances_Erev": np.empty(0, dtype=np.float32), # np.zeros(1, dtype=np.float32) + 120, #
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
dt = 0.1
duration = 3000

timer = time.time()
pyramidal = lib.ComplexNeuron(neuron["compartments"], neuron["connections"])
print('Creation object time ', time.time() - timer, ' sec')
pyramidal.integrate(dt, duration)
print('Simulation time ', time.time() - timer, ' sec')

Vsoma = pyramidal.getCompartmentByName('soma').getVhist()
Vdend = pyramidal.getCompartmentByName('dendrite').getVhist()

t = np.linspace(0, duration, Vsoma.size)

spike_rate = np.exp( (Vsoma - 30)/20 )

plt.plot(t, spike_rate)
plt.show()

plt.plot(t, Vsoma)
plt.plot(t, Vdend)
plt.show()