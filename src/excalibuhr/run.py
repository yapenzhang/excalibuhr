from excalibuhr import calib

workpath = './'
night_list = ['2023-02-27']

for night in night_list:
    ppl = calib.CriresPipeline(workpath, night=night, 
            num_processes=4, clean_start=False)
    # ppl.download_rawdata_eso(login='', prog_id='')
    ppl.run_recipes(
                    combine=True,
                    extract_2d=True,
    )
