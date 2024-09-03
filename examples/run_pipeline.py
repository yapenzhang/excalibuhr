if __name__ == '__main__':
    from excalibuhr import pipeline

    workpath = './'
    night_list = ['2023-02-26','2023-02-27']

    for night in night_list:
        
        ppl = pipeline.CriresPipeline(workpath, night=night, obs_mode='nod',
                num_processes=4, clean_start=False)
        
        # download data using astroquery package 
        # (or skip this if downloading data manually) 
        ppl.download_rawdata_eso(login='yzhang', prog_id='110.23RW.006', object='YSES1 bc')

        # reduce calibration files, calibrate science frames, 
        # and combine indvidual frames at each nodding position
        ppl.preprocessing() 

        # extract standard star spectrum
        ppl.obs_extract(object='zet CMa')
        
        # extract target spectrum
        ppl.obs_extract(object='YSES 1bc',
                        peak_frac={'A':0.5, 'B':0.95},
                        )
        
        # extract companion spectrum by giving either the location on the slit 
        # or the angular separation from the primary. The values don't have to very accurate.
        ppl.obs_extract(object='YSES 1bc',
                        savename='1b',
                        peak_frac={'A':0.3, 'B':0.8},
                        # companion_sep=1.7,
                        # remove_star_bkg = True,
                        )
        
        # refine wavelength solution with telluric transmission model and save extracted data
        ppl.refine_wlen_solution()

        # use the standard spectrum for spectrophotometry calibration 
        ppl.spec_response_cal(object='zet CMa', temp=15000, vsini=40, vsys=13.7)
        ppl.apply_correction()

        # use molecfit to correct for telluric absoprtion
        ppl.run_molecfit(object='YSES 1bc', wave_range=[(2.06,2.475)])
        
