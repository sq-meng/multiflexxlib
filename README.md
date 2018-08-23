# multiflexxlib
Tools library for inelastic neutron spectroscopy detector array MultiFLEXX.
## Introduction
`multiflexxlib` is a Python package for visualization and treatment of neuron spectroscopy data measured with cold neutron array detector MultiFLEXX. A detailed description on the detector can be found on the [HZB website](https://www.helmholtz-berlin.de/forschung/oe/em/transport-phenomena/em-amct-instruments/flex/multiflexx_en.html).

## Required Environment
`multiflexxlib` requires python3 version > 3.5. For installation of Python environment under Windows it is recommended to install a scientific Python package such as [Anaconda](https://www.anaconda.com/download/) to make your life easier.
> Support for MacOS is questionable at the moment due to peculiarities on how tkinter interacts with GUI routines of MacOS. The code should work though - and please report if it doesn't.  

Alternately, Python2 > 2.7 is partially supported. Most of the code is written with Python 2 compatibility in mind, but might contain (additional) bugs.

\>1GB of free memory recommended.

### Ask me anything!

If you are unsure about how to do something or are getting unexpected behaviour, open an issue or drop me an email!

## Installation
run command `pip install multiflexxlib` from windows or linux command console.

## Sample data
Data files for measurement of excitations in antiferromagnet MnF<sub>2</sub> can be found [here](https://github.com/sq-meng/multiflexxlib/tree/master/sampledata/MnF2). Please download the files into a folder.
## Usage
### Minimal Usage
Download the file [run.py](https://github.com/yumemi5k/multiflexxlib/blob/master/run.py) and save  somewhere. Double-click on saved file and select the folder containing your MultiFLEXX scan files when asked for data folder. All possible 2D const-E plots will be shown. Double-click on a plot to open the plot in its own window.

### Extended Usage
Please also refer to docstrings of classes and functions, accessible through `help()` function, e.g. `help(mfl.UBMatrix)`.

It is possible and recommended to use this package in an interactive Python interpreter such as `IPython` or a Matlab-esque Python development environment like `Spyder`, which both come with a default `Anaconda` installation.

`import multiflexxlib as mfl`

This is required before you can start using the package. `mfl` is a shorthand that you can also choose for yourself.
#### Loading data
`alldata = mfl.read_and_bin()`
>Alternately, use `alldata = mfl.read_and_bin(processes=4)` to load data with multiple CPU cores simultaneously. This does not always work.

>UB-matrix and plotting axes will be determined automatically if not provided. x and y axes are scaled to be equal in terms of absolute reciprocal length. Non-orthogonal axes is supported. If lattice parameter or plotting axes is to be overridden from scan files metadata, create a UBMatrix object as follows: `my_ubmatrix = mfl.UBMatrix([a, b, c, alpha, beta, gamma], hkl1, hkl2, plot_x, plot_y, a3_add, a4_add)` `hkl1, hkl2, plot_x, plot_y` should be given as 3-element list \[h, k, l\]. `plot_x` and `plot_y` parameters can be omitted if `plot_x == hkl1` and `plot_y == hkl2` `a3_add` and `a4_add` will be ADDED into A3 and A4 angles respectively. Pass this custom `UBMatrix` into `read_and_bin` as `mfl.read_and_bin(ub_matrix=my_ubmatrix)`

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

#### Plotting data
Let's suppose you want plots for data index number 0, 1, 2, 3 and 4:

`p = alldata.plot(subset=[0,1,2,3,4])`
>`[0,1,2,3,4]` here represents a `list` in Python. The `subset` parameter could be omitted if you want to plot all possible plots like in this case. A `Plot2D` object is returned to name p. A graph will be generated.
Plots can be panned and zoomed using controls in graph window. 

The returned `Plot2D` object can be used to access graph customization methods and 1D-cuts.

It is common for accidental Bragg scattering spurions to exhibit much higher intensity than inelastic signals. To crop out such spurions:
  
`p.set_plim(pmin=0, pmax=99.7)`

This limits dynamic range to 0.0% to 99.7%, ignoring the highest 0.3%.

The users are encouraged to explore other available methods for graphical customization. 


#### Cutting data
It might be interesting to do a 1D-cut on 1st and 3rd plots, which is done as follows:

`c = p.cut_voronoi([1, 1, 1], [1.1, 1.1, 1], subset=[0, 2])`

This does a cut through Voronoi partition of const-E data. Regions crossed by cut line are picked up and included in the cut. subset parameter can be omitted. `[1, 1, 1]` here is `[h, k, l]` values. `subset=[0, 2]` instead of `[1, 3]` because python index starts with `0`. This generates a cut from \[1, 1, 1\] to \[1.1, 1.1, 1\]. The `cut` method draws a line segment between specified cut start and end points, and each data point corresponding to crossed regions is subsequently projected onto cut axis.

It makes more sense to do a traditional cut with rectangular bins if amount of data is sufficiently high:

`c = p.cut_bins([1, 1, 1], [1.1, 1.1, 1], subset=[0, 2], no_points=21, ytol=[0, 0, 0.02])`

Users are strongly recommended to refer to docstring on how to define bin size.

To plot a Q-E dispersion plot:

`df.dispersion([1, 1, 1], [1.1, 1.1, 1], no_points=21)`

This generates a vertical stacking of 1D const-E cuts. It should be noted that data from all final energies is used, thus intensity might zigzag from resolution effects.

#### Exporting data

To export all data into CSV files:

`df.to_csv()`

This writes a series of CSV files into folder containing source scan files.