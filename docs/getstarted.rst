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
                                  workpath, night=night, 
                                  num_processes=4, clean_start=False
                                 )

.. note::

    ``num_processes`` sets the number of parallel processes when reducing science frames.
    ``clean_start`` gives the option to remove all previous products in case one needs to re-run the pipeline.

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


.. note::

    The ``combine`` parameter sets whether combining all frames during the night or processing each individual frame to keep the time resolusion.

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
        ppl.save_extracted_data()

The extracted 1d spectra will be saved as text files to the ``obs_calibrated`` folder.

If you need more customized reduction, please find the individual recipes explained in the
API reference of :class:`~excalibuhr.pipeline.CriresPipeline` class.


.. tip::

    * To extract spectra of spatially resolved planetary companions, set the ``companion_sep`` parameter to the angular separation of the planet in arcseconds. 
    
    * There are options to change the extraction aperture of the primary and companion (``aper_prim`` and ``aper_comp``). 
    
    * If one needs the 2-dimensional data for dedicated analysis (e.g. in the case of spatially resolved but non widely separated companions), set ``extract_2d=True`` to save the intermediate data product. 

    * If telluric standard stars have been observed during the night, then specify the object name with the ``std_object`` parameter to use it for the refinement of wavelength solution.

    * The pipeline can also call `Molecfit <https://www.eso.org/sci/software/pipelines/skytools/molecfit>`_ to correct for telluric absorptions if setting ``run_molecfit=True``.



