from excalibuhr import calib


ppl = calib.Pipeline('./', night='2021-10-16',
				key_gain='ESO DET CHIP GAIN',
				key_filename='ORIGFILE',
				key_dtype='ESO DPR TYPE',
				key_catg='ESO DPR CATG',
				key_DIT='ESO DET SEQ1 DIT',
				key_NDIT='ESO DET NDIT',
				key_wlen='ESO INS WLEN ID',
				key_nodpos='ESO SEQ NODPOS',
				key_nabcycle='ESO SEQ NABCYCLES',
				key_nodthrow='ESO SEQ NODTHROW',
				key_slitlen='ESO INS SLIT1 LEN',
				)
ppl.download_rawdata_eso(login='yzhang', prog_id='108.222Y.001')
ppl.extract_header()
ppl.cal_dark()
ppl.cal_flat_raw()
ppl.cal_flat_trace(debug=True)
ppl.cal_slit_curve(key_wave_min='ESO INS WLEN BEGIN', key_wave_max='ESO INS WLEN END', key_wave_cen='ESO INS WLEN CENY', debug=False)
ppl.cal_flat_norm(debug=True)
ppl.obs_nodding(debug=False)
ppl.obs_nodding_combine()
ppl.extract1d_nodding(companion_sep=1.7, debug=False)


