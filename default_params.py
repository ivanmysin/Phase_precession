
def default_param4optimization():
    default_param = {'precession_slope': 5, 'animal_velosity': 20, 'R_place_cell': 0.5, 'sigma_place_field': 4,
                     'theta_freq': 8, "use_x0": True, 'sigma_max_cm' : 30, }

    return default_param



def get_neuron_structure():
    soma_params = {
        "V0": -5.0,
        "C": 3.0,
        "Iextmean": 0.0,
        "Iextvarience": 1.5,
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

        "input_conduntance": None, # g_syn[soma_idxes, :],  # np.empty((0, 0), dtype=np.float32),
        "conduntances_Erev": None, # Erev[soma_idxes],  # np.empty(0, dtype=np.float32),
    }

    dendrite_params = {
        "V0": -5.0,
        "C": 3.0,
        "Iextmean": 0.0,
        "Iextvarience": 1.5, # 0.0001,
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

        "input_conduntance": None, # g_syn[dend_idxes, :],
        "conduntances_Erev": None, # Erev[dend_idxes],
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

    return neuron


def get_inputs_names():
    input_names = {
        'soma' : ['ca3', 'cck', 'pv', 'aac', 'bis'],
        'dend' : ['ec3', 'ngf', 'ivy', 'olm']
    }

    return input_names