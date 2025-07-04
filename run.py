from CP.CP import run_cp_and_save
from SMT.SMT import run_smt_and_save
#from MIP.MIP import run_mip_and_save
from sys import argv
from os import path
#import argparse

# parser = argparse.ArgumentParser(
#                     prog='CDMO',
#                     description='CDMO MCP solver')

BASE_PATH = path.dirname(__file__)

TYPE = argv[1] if len(argv) > 1 else 'CP'
DATA_INSTANCE_NUMBER = argv[2] if len(argv) > 2 else "all"
MINIZINC_EXECUTABLE = argv[3] if len(argv) > 3 else 'minizinc'

def run_instance(instance_number:int):
    print(f"Running instance {instance_number}...")
    if TYPE.lower() == 'cp':
        run_cp_and_save(
            instance_number,
            MINIZINC_EXECUTABLE,
            f"{BASE_PATH}/res/CP/{instance_number}.json"
        )
    elif TYPE.lower() == 'smt':
        run_smt_and_save(
            instance_number,
            f"{BASE_PATH}/res/SMT/{instance_number}.json"
        )
    elif TYPE.lower() == 'mip':
        run_mip_and_save(
            instance_number,
            f"{BASE_PATH}/res/MIP/{instance_number}.json"
        )
    else:
        raise ValueError(f"Unknown type: {TYPE}. Expected 'CP', 'SMT', or 'MIP'.")

if DATA_INSTANCE_NUMBER == "all":
    for i in range(1, 22):
        run_instance(i)
elif DATA_INSTANCE_NUMBER.index("-") > 0:
    min, max = DATA_INSTANCE_NUMBER.split("-")
    for i in range(int(min), int(max)):
        run_instance(i)
else:
    run_instance(int(DATA_INSTANCE_NUMBER))
    