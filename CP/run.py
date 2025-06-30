
import re
import json
import subprocess

##### CONFIGURATION #####

VARIANTS = {
    "gecode": {
        "mzn_file": "CP_model.mzn",
        "solver": "gecode",
    },
    "chuffed": {
        "mzn_file": "CP_model.mzn",
        "solver": "chuffed",
    },
    "3d_gecode": {
        "mzn_file": "cp_3d_variable_model.mzn",
        "solver": "gecode",
    },
    "3d_chuffed": {
        "mzn_file": "cp_3d_variable_model.mzn",
        "solver": "chuffed",
    },
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

def convert_minizinc_to_dict(input: str) -> dict:
    """
    Convert the output of the MiniZinc solver to a JSON format.
    Args:
        input (str): The output of the MiniZinc solver (see MINIZINC_OUT_SAMPLE_RAW for an example)
    Returns:
        A dictionary containing the output in the expected format (see MINIZINC_OUT_SAMPLE_DICT for an example)
    """

    # Extract the JSON part from the input string
    json_part = re.search(r'\{.*\}', input, re.DOTALL)
    if not json_part:
        raise ValueError("No JSON part found in the input string.")

    # Parse the JSON part
    dict_in = json.loads(json_part.group(0))
    dict_out = {
        'sol': dict_in.get('succ') or dict_in.get('u'),  # Assuming 'succ' contains the solution paths
        'obj': dict_in.get('_objective')  # Assuming '_objective' contains the objective value
    }

    # Extract time and objective value
    time_match = re.search(r'% time elapsed: (\d+\.\d+) s', input)
    if time_match:
        dict_out['time'] = float(time_match.group(1))
    else:
        raise ValueError("Time elapsed not found in the input string.")

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

def run_all_models(variants: dict, dzn_file:str, minizinc_path: str = "minizinc") -> dict:
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
        print(f"Running '{name}'")
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

##### EXECUTION #####

out = run_all_models(
    VARIANTS,
    "D:\\git\\CDMO\\CP\\minizinc_instances\\inst01.dzn",
    "D:\\Program Files\\MiniZinc\\minizinc.exe"
)
print("Execution completed. Results:\n", out)
save_to_file(out, "D:\\git\\CDMO\\CP\\results.json")