from glob import glob
import pandas as pd
import os
import json

for technique in ["CP", "SMT", "MIP"]:
    print(f"Reading {technique}")
    filepaths = glob(f'../res/{technique}/*.json') # get list of json files in folder
    df = pd.DataFrame(columns=["instance", "variant", "obj", "optimal", "time", "sol"])

    for f in filepaths:
        with open(f) as json_data:
            filename = os.path.basename(f).rsplit('.', 1)[0] # extract filename without extension
            instance_result = json.load(json_data)
            for variant in instance_result:
                row = pd.Series(instance_result[variant])
                row["instance"] = int(filename)
                row["variant"] = variant
                #print(row)
                df = pd.concat([
                    df,
                    row.to_frame().T
                ], ignore_index=True)
    df.astype({
            "instance":int, "variant":str, "obj":int, "optimal":bool, "time":int, "sol":str
        }).sort_values(["instance","variant"]).to_csv(f'{technique}.csv', index=False)
