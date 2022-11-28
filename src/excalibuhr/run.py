from excalibuhr import calib

keywords = {
    'key_filename': 'ORIGFILE',
    'key_target_name': 'OBJECT',
    'key_mjd': 'MJD-OBS',
    'key_ra': 'RA',
    'key_dec': 'DEC',
    'key_dtype':'ESO DPR TYPE',
    'key_catg': 'ESO DPR CATG',
    'key_DIT': 'ESO DET SEQ1 DIT',
    'key_NDIT': 'ESO DET NDIT',
    'key_wlen': 'ESO INS WLEN ID',
    'key_nodpos': 'ESO SEQ NODPOS',
    'key_nexp_per_nod': 'ESO SEQ NEXPO',
    'key_slitlen': 'ESO INS SLIT1 LEN',
    'key_slitwid': 'ESO INS SLIT1 NAME',
    'key_jitter': 'ESO SEQ JITTERVAL',
    # 'key_nabcycle': 'ESO SEQ NABCYCLES',
    # 'key_nodthrow': 'ESO SEQ NODTHROW',
}

workpath = './'
night_list = ['2021-10-16','2021-11-09']

for night in night_list:
    ppl = calib.Pipeline(workpath, night=night, **keywords)
    # ppl.download_rawdata_eso(login='yzhang', prog_id='108.222Y.001')
    ppl.extract_header()
    # ppl.cal_dark()
    # ppl.cal_flat_raw()
    # ppl.cal_flat_trace(debug=False)
    # ppl.cal_slit_curve(
    #                     key_wave_min='ESO INS WLEN BEGIN', 
    #                     key_wave_max='ESO INS WLEN END', 
    #                     key_wave_cen='ESO INS WLEN CENY', debug=True)
    # ppl.cal_flat_norm(debug=False)
    # ppl.obs_nodding(debug=True)
    # ppl.obs_nodding_combine()
    # ppl.extract1d_nodding(# f_star={'A': 0.295, 'B': 0.795}, \
    #                         companion_sep=1.8, debug=False)
    # ppl.run_skycalc()
    # ppl.refine_wlen_solution(debug=True)
    # ppl.save_extracted_data(debug=True)

# calib.CombineNights(workpath, night_list)