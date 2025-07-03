from CP.run import run_and_save, VARIANTS
from sys import argv

TYPE = argv[1] if len(argv) > 1 else 'CP'
MINIZINC_EXECUTABLE = argv[2] if len(argv) > 2 else 'minizinc'
DATA_FILE = argv[3] if len(argv) > 3 else 'CP/minizinc_instances/inst01.dzn'

if TYPE.lower() == 'CP':
    if DATA_FILE == "all":
        print("TODO run all data files")
    else:
        run_and_save(VARIANTS, MINIZINC_EXECUTABLE, DATA_FILE)
elif TYPE.lower() == 'SMT':
    print("TODO SMT")
elif TYPE.lower() == 'MIP':
    print("TODO MIP")
else:
    raise ValueError(f"Unknown type: {TYPE}. Expected 'CP', 'SMT', or 'MIP'.")