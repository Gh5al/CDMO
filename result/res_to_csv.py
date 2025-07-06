from glob import glob
import pandas as pd
import os
import json

for technique in ["CP", "SMT", "MIP"]:
    print(f"Reading {technique}")
    filepaths = glob(f'../res/{technique}/*.json') # get list of json files in folder
    df = pd.DataFrame()

    for f in filepaths:
        with open(f) as json_data:
            filename = os.path.basename(f).rsplit('.', 1)[0] # extract filename without extension
            instance_result = json.load(json_data)
            row_dict = {"instance":int(filename)}
            for variant in instance_result.keys():
                row_dict[variant] = instance_result[variant]["obj"]
                #row[f'{variant}_optimal'] = instance_result[variant]["optimal"]
            row_df = pd.DataFrame([row_dict])
            print(row_df)
            df = pd.concat([df, row_df], ignore_index=True)
    df.astype({"instance":int })\
        .sort_values(["instance"])\
        .to_csv(f'{technique}.csv', index=False)
