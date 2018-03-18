# multiflexxlib
Tools library for inelastic neutron spectroscopy detector array MultiFLEXX.
## Introduction
`multiflexxlib` is a Python package for visualization and treatment of neuron spectroscopy data from cold neutron continous angle multiple energy analysis (CAMEA) array detector MultiFLEXX. A detailed description on the detector can be found on [HZB website](https://www.helmholtz-berlin.de/forschung/oe/em/transport-phenomena/em-amct-instruments/flex/multiflexx_en.html)

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
>Alternately, use `alldata = mfl.read_and_bin(processes=4)` to load data with multiple CPU cores simultaneously. This does not always work.

You will be asked for a folder containing data like in minimal usage. All data from the folder will be loaded, scans done under identical conditions will be binned together.

`print(alldata)`

This prints a tabular summary.

| |ei|en|tt|mag|locus_a|locus_p|points|
|----|----|----|---|---|----|---|---|
|0|8.000747|3.500747|294.8735|0|1p 180v|1p 180v|3782 pts|
|1|8.000747|4.000747|294.8735|0|1p 180v|1p 180v|3538 pts|
|2|8.000747|4.500747|294.8735|0|1p 180v|1p 180v|3538 pts|
|3|8.000747|5.000747|294.8735|0|1p 180v|1p 180v|3660 pts|
|4|8.000747|5.500747|294.8735|0|1p 180v|1p 180v|3782 pts|

>The binning process considers any two values that are different by less than tolerance threshold to be identical. Defaults are energy: 0.05meV, temperature: 1K, angles 0.2&deg; and magnetic field 0.05T. Partially and fully repeating and overlapping scans will be dealt with in a sensible manner.

Let's suppose you want plots for data index number 0, 1, 2, 3 and 4:

`p = alldata.plot(select=[0,1,2,3,4])`
>`[0,1,2,3,4]` here represents a `list` in Python. The `select` parameter could be omitted if you want to plot all possible plots like in this case. A `Plot2D` object is returned to name p.
A graph as follows will be generated:

`Insert graph here`

Plots can be panned and zoomed using controls in graph window. UB-matrix and plotting axes will be determined automatically if not provided. x and y axes are scaled to be equal in terms of absolute reciprocal length.
>Alternatively, if crystal lattice information or scattering plane definition in data files is inaccurate, you can create a custom UB-matrix object and pass it as a parameter to `read_and_bin`. See documentation for details. 

The returned `Plot2D` object can be used to access 

It might be interesting to do a 1D-cut on 1st and 3rd plots, which is done as follows:

`c = p.cut([1, 1, 1], [1.1, 1.1, 1], select=[0, 2])`

select parameter can be omitted. `[1, 1, 1]` here is `[h, k, l]` values. `select=[0, 2]` instead of `[1, 3]` because python index starts with `0`.

