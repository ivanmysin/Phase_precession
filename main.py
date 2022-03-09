import numpy as np
import pandas as pd
import optimization_and_run as optlib
from default_params import default_param4optimization
from presimulation_lib import get_soma_dend_idxes

output_path = "./output/"
num = 'default___'
############################################
datafile = "inputs_data.csv"
data = pd.read_csv(datafile, header=0, comment="#", index_col=0)

param = default_param4optimization()
optlib.optimization_model(num, param, data, output_path)

# s, d = get_soma_dend_idxes(data)
# print(s)
# print(d)