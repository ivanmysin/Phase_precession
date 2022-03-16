import numpy as np
import pandas as pd
import optimization_and_run as optlib
from default_params import default_param4optimization
from copy import deepcopy
import os
import shutil
import h5py
#############################################################################################
def research_optim_results(sourse_param, output_path, source_file, theta_freqs, animal_velosities):
    params_list = []
    for theta_freq in theta_freqs:
        for animal_velosity in animal_velosities:
            param = deepcopy(sourse_param)
            param['theta_freq'] = theta_freq
            param['animal_velosity'] = animal_velosity
            params_list.append(param)

    optlib.multipal_run(output_path, params_list, data, source_file)
##########################################################################################
def estimate_weights_significance(sourse_param, output_path, source_file, theta_freqs, animal_velosities, data):
    weights = np.around( np.arange(0, 1.05, 0.05), 3)
    for pop_idx, pop_name in enumerate(data.columns):
        output_path_pop = output_path + pop_name + '/'
        try:
            os.mkdir(output_path_pop)
        except FileExistsError:
            pass

        for w in weights:
            output_path_pop_w  = output_path_pop + str(w) + '/'
            try:
                os.mkdir(output_path_pop_w)
            except FileExistsError:
                pass

            file_name = os.path.basename(source_file)
            file_name = 'w_' + str(w) + file_name
            shutil.copyfile(source_file, output_path_pop_w + file_name)
            with h5py.File(output_path_pop_w + file_name, "a") as hdf_file:
                hdf_file['Weights'][pop_idx] = w
            output_path_pop_w_research = output_path_pop_w + 'research/'
            try:
                os.mkdir(output_path_pop_w_research)
            except FileExistsError:
                pass
            source_path_file_wpop = output_path_pop_w + file_name
            research_optim_results(sourse_param, output_path_pop_w_research, source_path_file_wpop, theta_freqs, animal_velosities)


    return

##########################################################################################

output_path = "./output/default_optimization/"
num = 'default_experiment.hdf5'
############################################
datafile = "inputs_data.csv"
data = pd.read_csv(datafile, header=0, comment="#", index_col=0)

default_param = default_param4optimization()
# optlib.optimization_model(num, default_param, data, output_path)

# research optimized model
source_file = output_path + num
# output_path_research = output_path + 'research/'

theta_freqs = [4, 6, 8, 10, 12]
animal_velosities = [10, 15, 20, 25, 30]

# research_optim_results(default_param, output_path_research, source_file, theta_freqs, animal_velosities)

output_path_weights = output_path + 'weights/'
estimate_weights_significance(default_param, output_path_weights, source_file, theta_freqs, animal_velosities, data)