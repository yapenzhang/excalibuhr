# excalibuhr

A Python package for data reduction of high-resolution spectroscopy of exoplanets and brown dwarfs. It is designed for VLT/CRIRES+, but will be extended to other similar instruments such as Keck/NIRSPEC.


## Installation

```
python -m pip install -e .
```


## Quick start

*excalibuhr* is an end-to-end pipeline for extracting high-resolution spectra from VLT/CRIRES+ data. An example of running the pipeline can be found in `src/excalibuhr/run.py`. 

The organization of the reduction folder is on a nightly basis. Multiple targets observed in the same night can be processed altogether and share the same calibration data. 
To begin with, chose any working directory you like, create or copy the `run.py` script, create a folder named by the night of observation, e.g. `2023-03-03`, and download the raw science data and associated calibration data to a folder named `raw` inside the night folder. Then run the pipeline simply by executing the script.

The main steps in the pipeline include pre-processing (such as dark correction, flat fielding, order tracing, slit-tilt tracing, and background subtraction), spectrum extraction, and post-processing (such as wavelength calibration, telluric correction). The processed calibration data and products are saved in the `cal` and `out` folders. It also generates standard plots for visual checks of the results.


## Documentation

Documentation can be found at 