# multiflexxlib
Tools library for inelastic neutron spectroscopy detector array MultiFLEXX.
## Introduction
`multiflexxlib` is a Python package for the visualization and treatment of neuron spectroscopy data from cold neutron continous angle multiple energy analysis (CAMEA) array detector MultiFLEXX. A detailed description on the detector can be found on [HZB website](https://www.helmholtz-berlin.de/forschung/oe/em/transport-phenomena/em-amct-instruments/flex/multiflexx_en.html)

## Required Environment
This package requires python3 version > 3.5. For installation of Python environment under Windows it is recommended to install a scientific Python package such as [Anaconda](https://www.anaconda.com/download/) to skip the installation of some tricky packages like `numpy`.

\>1GB of free memory is recommended.
## Installation
run command `pip install multiflexxlib` in windows command console or linux console.
## Usage
### Minimal Usage
Download file [run.py](https://github.com/yumemi5k/multiflexxlib/blob/master/run.py) and save  somewhere. Double-click on saved file and select the folder containing only your MultiFLEXX scan files when asked for data folder. All possible 2D const-E plots will be shown.
### Extended Usage
Although this section is written with users less familiar with Python in mind, a minimal knowledge of Python language is recommended.

It is possible and recommended to use this package in an interactive Python interpreter such as `IPython` or a Matlab-esque Python development environment like `Spyder`, which both come with default `Anaconda` installation.

import multiflexxlib as mfl`

This is required before you can start using the package. ```mfl``` is just a shorthand which you can also choose for yourself.

`alldata = mfl.read_and_bin()`
> Alternately, use `alldata = mfl.read_and_bin(processes=4)` to load data with multiple CPU cores simultaneously. This does not always work.

You will be asked for a folder containing data like in minimal usage.

`print(alldata)`

This prints a tabular summary.

ei|en|tt|mag|locus_a|locus_p|points
0|8.000747|3.500747|294.8735|0|1p|180v|1p|180v|3782|pts
1|8.000747|4.000747|294.8735|0|1p|180v|1p|180v|3538|pts
2|8.000747|4.500747|294.8735|0|1p|180v|1p|180v|3538|pts
3|8.000747|5.000747|294.8735|0|1p|180v|1p|180v|3660|pts
4|8.000747|5.500747|294.8735|0|1p|180v|1p|180v|3782|pts
