# excalibuhr

A Python package for data reduction of high-resolution spectroscopy of exoplanets and brown dwarfs. It has an end-to-end pipeline for VLT/CRIRES+.

## Documentation

### Installation

Download the repository and install the package by running pip in the local repository folder with the following command

``` shell
python -m pip install -e .
``` 

### Quick start

Create a working directory named by the observing night, e.g. `2024-06-06`. Download the raw science data and calibration data taken with VLT/CRIRES+ and put them all to the raw folder inside the nightly working directory. Then run the pipeline as follows.

``` python
from excalibuhr import pipeline
workpath = './'
night = '2023-06-06'
ppl = pipeline.CriresPipeline(workpath, night=night, obs_mode='nod', num_processes=4)
ppl.run_recipes(combine=True)
```

Detailed documentation and customized usages can be found at [https://excalibuhr.readthedocs.io/](https://excalibuhr.readthedocs.io/).

## Attribution

Please cite [Zhang et al. (2024)](http://arxiv.org/abs/2409.16660) if *excalibuhr* is used in your research.

## Contributor

Yapeng Zhang

Sam de Regt

Darío González Picos


## Contact

Contributions, feature requests, or bug reports are welcome through [Github page](https://github.com/yapenzhang/excalibuhr) or email to yapzhang@caltech.edu 

## References

* [pycrires](https://github.com/tomasstolker/pycrires)

* [cr2res pipeline](https://www.eso.org/sci/software/pipelines/cr2res/cr2res-pipe-recipes.html)

