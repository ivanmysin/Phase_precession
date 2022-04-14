# Phase_precession
Code for simulation inhibitory control of place cell phase precession in the CA1 field


Dependencies
-----------------------------------
Model is runned under Python 3.7. OS is Ubuntu 18.04 or Ubuntu 20.04, another os in not tested.

We will need to install git, pip:

    sudo apt update
    sudo apt install git python3-pip


Python packages for simulations:
* Cython
* Numba

Packages for saving, processing and plotting:
* numpy, scipy, matplotlib
* h5py


Installation pakages via pip:
      
    sudo pip3 install numpy scipy matplotlib numba h5py cython

Directories and files
-----------------------------------
     /output - directory for saving simulation results and plots
     inputs_data.csv - file with parameters of the inputs
     main.py - main python file
     default_params.py - python file containing default parameters of simulation
     presimulation_lib.py - helper functions for calculating some simulation parameters
     neuron_simulation_lib.pyx - contains code for neuron simulation
     processing.py - library of funtions, which are used for processing
     figure.py - file, which are plot figures for article.
     setup.py - file for compilation of cython code
     

How to run
-----------------------------------
You need:
* Clone this repository.
Open terminal in working directory and execute commands:

      git clone https://github.com/ivanmysin/Phase_precession


* Compile *neuron_simulation_lib.pyx* file 

      bash setup.sh

or 

    python3 setup.py build_ext --inplace


* Run in terminal:
  
        python3 main.py
  
Simulation results are saved to hdf5 file.
*python3* is the default command for calling the python 3 interpreter, however, it may be different on your system.
Substitute the call of the interpreter for which you have installed dependencies.

* To process the results, run *process.py*.  It will process and save
in the same file, wavelet spectra, bands, distribution of neurons by rhythm phases, etc.
  
      python3 process.py

* Run *figure.py* file to plot figure

      python3 figure.py



Structure of HDF5 file with results
-----------------------------------
You can use any free convenient program for viewing hdf5 files,
for example, HDF COMPASS or hdfview.

Path for reading datasets from hdf5 file:

somatic intracellular potential: /V

simulated spike rate: /spike_rate

target (theoretical) spike rate: /teor_spike_rate

Results of optimization:

/Weights

/Centers

/Sigmas

Centers and Sigmas are saved in ms. Parameters of each simulations are kept in atributes of the file. 



