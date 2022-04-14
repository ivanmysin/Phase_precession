import numpy as np
import pandas as pd
import optimization_and_run as optlib
from default_params import default_param4optimization
import presimulation_lib as plib
from copy import deepcopy
import os
import shutil
import h5py


#############################################################################################
def research_optim_results(sourse_param, output_path, source_file, theta_freqs, animal_velosities, data):
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
        
        # if str(pop_name) != 'pv' and str(pop_name) != 'cck':
        #     continue
        print("Population name ", pop_name)
        
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
            file_name = pop_name + '_w_' + str(w) + '_' + file_name
            shutil.copyfile(source_file, output_path_pop_w + file_name)
            with h5py.File(output_path_pop_w + file_name, "a") as hdf_file:
                hdf_file['Weights'][pop_idx] = w
            output_path_pop_w_research = output_path_pop_w + 'research/'
            try:
                os.mkdir(output_path_pop_w_research)
            except FileExistsError:
                pass
            source_path_file_wpop = output_path_pop_w + file_name
            
            research_optim_results(sourse_param, output_path_pop_w_research, source_path_file_wpop, theta_freqs, animal_velosities, data)



    return

##########################################################################################
def main():
    theta_freqs = [4, 6, 8, 10, 12]
    animal_velosities = [10, 15, 20, 25, 30]
    precession_slope = [2.5, 7]
    sigma_place_field = [2, 5]
    
        
    datafile = "inputs_data.csv"
    data = pd.read_csv(datafile, header=0, comment="#", index_col=0)
    data.loc["phi"] = np.deg2rad(data.loc["phi"])
    data.loc["kappa"] = [plib.r2kappa(r) for r in data.loc["R"]]
    
    ############################################


    output_path = "./output/default_optimization/" # "./output/small_sigmas/"  #
    num = 'default_experiment.hdf5'  # 'experiment_small_sigmas.hdf5'    # 

    default_param = default_param4optimization()
    optlib.optimization_model(num, default_param, data, output_path)

    
    # research optimized model
    source_file = output_path + num
    output_path_research = output_path + 'research/'
    try:
        os.mkdir(output_path_research)
    except FileExistsError:
        pass

    research_optim_results(default_param, output_path_research, source_file, theta_freqs, animal_velosities, data)

    output_path_weights = output_path + 'weights/'
    try:
        os.mkdir(output_path_weights)
    except FileExistsError:
        pass
    estimate_weights_significance(default_param, output_path_weights, source_file, theta_freqs, animal_velosities, data)
    

    output_path4multipal_optimization = './output/multipal_optimization/'
    # './output/small_sigma_multipal_optimization/'
    try:
        os.mkdir(output_path4multipal_optimization)
    except FileExistsError:
        pass
    
    
    # multipal run of optimizarion
    for idx in range(1, 30):
        filename = str(idx)
        param = deepcopy(default_param)
        param["use_x0"] = False

        param['animal_velosity'] = np.random.uniform(animal_velosities[0], animal_velosities[-1])
        param['theta_freq'] = np.random.uniform(theta_freqs[0], theta_freqs[-1])
        param['precession_slope'] = np.random.uniform(precession_slope[0], precession_slope[-1])
        param['sigma_place_field'] = np.random.uniform(sigma_place_field[0], sigma_place_field[-1])
        optlib.optimization_model(filename, param, data, output_path4multipal_optimization)
    

    
   
    
    
    
main()


