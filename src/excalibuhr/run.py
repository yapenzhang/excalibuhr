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
    'key_nabcycle': 'ESO SEQ NABCYCLES',
    # 'key_nodthrow': 'ESO SEQ NODTHROW',
    'key_wave_min': 'ESO INS WLEN BEGIN', 
    'key_wave_max': 'ESO INS WLEN END', 
    'key_wave_cen': 'ESO INS WLEN CENY', 
}

workpath = './'
night_list = ['2021-10-16','2021-11-09']

for night in night_list:
    ppl = calib.Pipeline(workpath, night=night, **keywords)
    # ppl.download_rawdata_eso(login='yzhang', prog_id='108.222Y.001')
    ppl.run_recipes(companion_sep=1.8)

# calib.CombineNights(workpath, night_list)