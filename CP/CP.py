
import re
import json
import math
import subprocess
from sys import argv
from os import path

##### CONFIGURATION #####

BASE_PATH = path.dirname(__file__)
VARIANTS = {
    "gecode": {
        "mzn_file": BASE_PATH+"/CP_model.mzn",
        "solver": "gecode",
    },
    "chuffed": {
        "mzn_file": BASE_PATH+"/CP_model.mzn",
        "solver": "chuffed",
    },
    "gecode_search": {
        "mzn_file": BASE_PATH+"/CP_model_search.mzn",
        "solver": "gecode",
    },
    # "lns_chuffed": {
    #     "mzn_file": BASE_PATH+"/CP_model_LNS.mzn",
    #     "solver": "chuffed",
    # },
}

##### SAMPLE DATA #####

SAMPLE_CMD = "D:\\Program Files\\MiniZinc\\minizinc.exe --solver gecode --output-mode json --output-time --solver-time-limit 300000 --output-objective D:\\git\\CDMO\\CP\\CP_model.mzn D:\\git\\CDMO\\CP\\minizinc_instances\\inst01.dzn"

MINIZINC_OUT_SAMPLE_RAW = """
{
  "succ" : [[7, 2, 1, 3, 5, 6, 4], [1, 5, 3, 4, 6, 7, 2]],
  "tot_dist" : [14, 14],
  "max_distance" : -2147483646,
  "_objective" : 14
}
% time elapsed: 0.63 s
----------
==========
% time elapsed: 0.64 s
"""

MINIZINC_OUT_SAMPLE_DICT =  {
 "time": 1,
 "optimal": True,
 "obj": 14,
 "sol" : [[7, 2, 1, 3, 5, 6, 4], [1, 5, 3, 4, 6, 7, 2]],
}

##### UTILITY FUNCTIONS #####

def convert_minizinc_to_dict(minizinc_output: str) -> dict:
    """
    Convert the output of the MiniZinc solver to a JSON format.
    Args:
        minizinc_output (str): The output of the MiniZinc solver (see MINIZINC_OUT_SAMPLE_RAW for an example)
    Returns:
        A dictionary containing the output in the expected format (see MINIZINC_OUT_SAMPLE_DICT for an example)
    """

    # Extract the JSON part from the minizinc_output string
    json_part = re.search(r'\{.*\}', minizinc_output, re.DOTALL)
    if not json_part and "=====UNKNOWN=====" in minizinc_output:
        return {
            "time": 300,
            "optimal": False,
            "obj": -1,
            "sol": []
        }
    elif not json_part:
        raise ValueError("No JSON part found in minizinc_output:\n", minizinc_output)
    
    # Parse the JSON part
    dict_in = json.loads(json_part.group(0))
    solution_raw = dict_in.get('succ') or dict_in.get('u'),
    try:
        #m = len(solution_raw[0])
        n = len(solution_raw[0][0])-1
        #print(m, n)
        sol = []
        #items = list(range(n))
        # extract solution for successor approach
        for courier in solution_raw[0]:
            sub_sol = []
            prev = courier[n]
            while (prev != n+1):
                for i in range(n):
                    if i+1 == prev:
                        sub_sol.append(i + 1)
                        prev = courier[i]
                        break
            sol.append(sub_sol)

        print(f"Solution: {solution_raw} -> {sol}")
    except Exception as e:
        print(f"Error processing solution:", e)
        sol = solution_raw  # Fallback to raw solution if processing fails

    dict_out = {
        'sol': sol,
        'obj': dict_in.get('_objective')  # Assuming '_objective' contains the objective value
    }

    # Extract time and objective value
    time_match = re.search(r'% time elapsed: (\d+\.\d+) s', minizinc_output)
    if time_match:
        dict_out['time'] = min(300, math.floor(float(time_match.group(1))))
    else:
        raise ValueError("Time elapsed not found in minizinc_output.")

    dict_out['optimal'] = dict_out.get("time") < 300
    return dict_out

def run_single_model(mzn_file:str,
                     dzn_file:str,
                     solver:str="gecode",
                     time_limit:int=300000,
                     minizinc_path:str="minizinc") -> str:
    """
    Execute the MiniZinc model with the given parameters and return the output
    
    Args:
        mzn_file (str): Path to the MiniZinc model file.
        dzn_file (str): Path to the MiniZinc data file.
        solver (str): The solver to use (default is 'gecode').
        time_limit (int): Time limit for the solver in milliseconds (default is 300000).
        minizinc (str): Path to the MiniZinc executable (default is 'minizinc').
    
    Returns:
        The output of the MiniZinc solver
    """

    cmd = [
        minizinc_path,
        "--solver", solver,
        "--output-mode", "json",
        "--output-time",
        "--output-objective",
        "--solver-time-limit", str(time_limit),
        mzn_file,
        dzn_file
    ]

    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"MiniZinc execution failed: {result.stderr}")

    return result.stdout

def run_all_models(variants: dict, dzn_file:str, minizinc_path: str) -> dict:
    """
    Execute all models defined in the variants dictionary and return their outputs.
    
    Args:
        variants (dict): Dictionary containing model configurations.
        dzn_file (str): Path to the MiniZinc data file.
        minizinc_path (str): Path to the MiniZinc executable (default is 'minizinc').
    
    Returns:
        A dictionary with model names as keys and their outputs as values.
    """
    results = {}
    for name, config in variants.items():
        print(f"Running '{name}' on '{dzn_file}'")
        try:
            output = run_single_model(
                mzn_file=config['mzn_file'],
                dzn_file=dzn_file,
                solver=config['solver'],
                time_limit=config.get('time_limit', 300000),
                minizinc_path=minizinc_path
            )
            results[name] = convert_minizinc_to_dict(output)
            print(f"Executed '{name}' successfully.")
        except Exception as e:
            print(f"Error executing '{name}': '{e}'")
    return results

def save_to_file(out_dict: dict, file_path: str):
    """
    Save the output dictionary to a JSON file.
    
    Args:
        out_dict (dict): The output dictionary to save.
        file_path (str): The path to the file where the output will be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(out_dict, f, indent=2)
    print(f"Results saved to {file_path}")

def check_executable(executable: str):
    try:
        subprocess.run([executable, '--version'])
    except Exception as e:
        raise RuntimeError(f"Executable '{executable}' not found") from e;

def run_cp_and_save(data_instance_number:int, minizinc_path: str, result_file: str):
    check_executable(minizinc_path)
    dzn_file = BASE_PATH + f"/minizinc_instances/inst{'{:0>2}'.format(data_instance_number)}.dzn"
    out = run_all_models(VARIANTS, dzn_file, minizinc_path)
    #print("Execution completed. Results:\n", out)
    save_to_file(out, result_file)

##### EXECUTION #####

if __name__ == '__main__':
    DATA_INSTANCE_NUMBER = argv[1] if len(argv) > 1 else 1
    MINIZINC_EXECUTABLE = argv[2] if len(argv) > 2 else 'minizinc'
    result_file = BASE_PATH + f"/../res/CP/{DATA_INSTANCE_NUMBER}.json"
    run_cp_and_save(int(DATA_INSTANCE_NUMBER), MINIZINC_EXECUTABLE, result_file)