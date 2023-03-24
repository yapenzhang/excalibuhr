from excalibuhr import calib

workpath = './'
night_list = ['2023-02-27']

for night in night_list:
    ppl = calib.CriresPipeline(workpath, night=night, 
            num_processes=4, clean_start=False)
    # ppl.download_rawdata_eso(login='', prog_id='')
    ppl.preprocessing()
    ppl.obs_extract(object='V GQ Lup', 
                    # companion_sep=0.71,
    )
    ppl.refine_wlen_solution(run_skycalc=True, debug=True)
    ppl.save_extracted_data()

# calib.CombineNights(workpath, night_list)