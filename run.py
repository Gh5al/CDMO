from CP.run import run_and_save
from sys import argv
#import argparse

# parser = argparse.ArgumentParser(
#                     prog='CDMO',
#                     description='What the program does',
#                     epilog='Text at the bottom of help')

TYPE = argv[1] if len(argv) > 1 else 'CP'
DATA_INSTANCE_NUMBER = argv[2] if len(argv) > 2 else "all"
MINIZINC_EXECUTABLE = argv[3] if len(argv) > 3 else 'minizinc'

if TYPE.lower() == 'cp':
    if DATA_INSTANCE_NUMBER == "all":
        for i in range(1, 22):
            run_and_save(i, MINIZINC_EXECUTABLE)
    else:
        run_and_save(int(DATA_INSTANCE_NUMBER), MINIZINC_EXECUTABLE)
elif TYPE.lower() == 'smt':
    print("TODO SMT")
elif TYPE.lower() == 'mip':
    print("TODO MIP")
else:
    raise ValueError(f"Unknown type: {TYPE}. Expected 'CP', 'SMT', or 'MIP'.")