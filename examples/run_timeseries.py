# Example: reduce a time series of CRIRES+ nodding observations.
#
# In time-series mode every individual exposure is extracted separately
# (rather than combining the frames at each nodding position). The
# per-exposure spectra are then assembled by `save_data` into a single FITS
# file holding the flux/error together with the MJD and airmass of
# each exposure.
#
from excalibuhr import pipeline


def main():

    workpath = './'
    night_list = ['2022-03-30']

    for night in night_list:

        ppl = pipeline.CriresPipeline(workpath, night=night, obs_mode='nod',
                num_processes=4, clean_start=False)

        # reduce calibration files and calibrate the science frames
        ppl.run_recipes(combine=False, extract_2d=False)

        # [optional] refine the wavelength solution with a telluric model
        ppl.refine_wlen_solution()

        # assemble the per-exposure spectra into a single time-series FITS file:
        #   FLUX, FLUX_ERR : (n_exposure, n_order, n_pixel)
        #   WAVE           : (n_order, n_pixel)
        #   MJD, AIRMASS   : (n_exposure,)
        # the primary header is taken from the first (earliest) exposure.
        ppl.save_data()


if __name__ == '__main__':
    main()
