.. _getstarted:

Get started
###########

*excalibuhr* is an end-to-end pipeline for extracting high-resolution spectra designed for VLT/CRIRES+. 
The organization of the reduction folder is on a nightly basis. Multiple targets (such as the telluric standard star) observed in the same night can be processed altogether. 
Within the folder of individual night, there are the ``raw``, ``cal``, and ``out`` folders containing the raw observation and calibration files, the processed calibratoion files, 
and the processed science data, respectively.


Initialize pipeline
*******************

To begin with, let's initialize the pipeline with the working directory and the observing night.

.. code-block:: python 

    from excalibuhr import pipeline

    workpath = './'
    night = '2023-06-06'

    ppl = pipeline.CriresPipeline(
                                  workpath, night=night, obs_mode='nod',
                                  num_processes=4, clean_start=False
                                 )

The ``num_processes`` sets the number of parallel processes when reducing science frames.
The ``clean_start`` gives the option to remove all previous products in case one needs to re-run the pipeline.

Download data
*************

Then download the raw science data and associated calibration data from `ESO Science Archive <http://archive.eso.org/cms.html>`_ , and put them all to the ``raw`` folder inside the night folder.

Alternatively, the pipeline also has a method for downloading the data using the ``astroquery`` package. 

.. code-block:: python 

    username = 'yzhang' #your ESO username
    program_id = '107.22TG' 

    ppl.download_rawdata_eso(login=username, prog_id=program_id)


Run pipeline
************

The main steps in the pipeline include organizing raw data, pre-processing (such as dark correction, flat fielding, order tracing, slit-tilt tracing, and background subtraction), 
spectrum extraction, post-processing (such as wavelength calibration, telluric correction), and saving the output data. It also generates standard plots for visual checks of the results.

To run the whole chain of recipes in the pipeline:

.. code-block:: python 

    ppl.run_recipes(combine=True) 

The ``combine`` parameter sets whether combining all frames at each nodding position or processing each individual frame to keep the time resolusion.

This function essentially runs the following recipes:

.. code-block:: python 

        ppl.extract_header()
        ppl.cal_dark()
        ppl.cal_flat_raw()
        ppl.cal_flat_trace()
        ppl.cal_slit_curve()
        ppl.cal_flat_norm()
        ppl.obs_nodding()
        ppl.obs_nodding_combine() #optional
        ppl.obs_extract()
        ppl.refine_wlen_solution()

If you need more customized reduction, please find the individual recipes explained in the
API reference of :class:`~excalibuhr.pipeline.CriresPipeline` class.

.. note::

    * To extract spectra of spatially resolved planetary companions with the ``obs_extract`` recipe, one can either set the ``companion_sep`` parameter to the angular separation of the planet in arcseconds or provide the location on the slit with e.g. ``peak_frac={'A':0.3, 'B':0.8}``. These values don't have to very accurate. 
    
    * If telluric standard stars have been observed during the night, we can use it for the refinement of wavelength solution and instrumental response calibration. In the ``run_recipes`` method, specify the ``std_prop`` as a dictionary containing the name, teff, vsini, and rv of the standard star.

    * The pipeline can also call `Molecfit <https://www.eso.org/sci/software/pipelines/skytools/molecfit>`_ to correct for telluric absorptions if setting ``run_molecfit=True``.



Access the intermediate data product
************************************

The calibrated 2-dimensional images is save to ``.npz`` files and can be simply loaded with the following code for dedicated analysis on off-axis spectral data.

.. code-block:: python

    from excalibuhr.data import DETECTOR

    extr2d = DETECTOR(filename='the_2d_product_file.npz')

    flux2d = extr2d.flux 
    variance2d = extr2d.var 
    psf2d = extr2d.psf 

    # These nested list contains the 2d images per detector and order. 
    # The shape is (detector x order x spacial pixels x spectral channels).


