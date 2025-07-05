# %% [markdown]
# <a href="https://colab.research.google.com/github/Gh5al/CDMO/blob/main/MIP/MIP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# %% [markdown]
# 3d decision variable + MTZ constraint to prevent subtours<br>
# 3 solvers are used:
# - GLPK
# - CBC
# - HiGHS

# %%
from pyomo.environ import ConcreteModel, Set, Constraint, ConstraintList, Var, value, Objective, minimize, SolverFactory, Binary, PositiveIntegers
from pyomo.contrib.appsi.solvers import Cbc, Highs
import time
import math
import re
import os
import json

# %%
def read_data(filename):
  dist = []
  with open(filename,'r') as f:
    m = int(f.readline().strip())
    n = int(f.readline().strip())
    capacity = [int(s) for s in f.readline().strip().split()]
    size = [int(s) for s in f.readline().strip().split()]
    for i in range(n+1):
      dist.append([int(s) for s in f.readline().strip().split()])
  return m,n,capacity,size,dist

def sol_to_dict(t,obj,sol):
  if t<300:
    optimal = "true"
    time = math.floor(t)
  else:
    optimal = "false"
    time = 300
  # Create JSON structure
  return {
          "time": time,
          "optimal": optimal,
          "obj": int(obj),
          "sol": sol
  }

# %%
def build_model_from_data_file(file_path:str):
  m,n,capacity,size,dist = read_data(file_path)
  print(f"couriers: {m}, items: {n}")


  #3d decision variable xkij, if the courier k goes from i to j = 1 otherwise 0 and MTZ constraint to prevent subtours
  model = ConcreteModel()
  model.K = Set(initialize=range(m)) #couriers
  model.I = Set(initialize=range(n)) #items
  model.loc=Set(initialize=range(n+1)) #locations

  #variables
  model.x = Var(model.K, model.loc,model.loc,domain=Binary) #decision variable
  model.o = Var(model.K, model.I,domain=PositiveIntegers)  # order variable to prevent subtour
  model.dist = Var(model.K, domain=PositiveIntegers) #memorize the distance for each courier
  model.obj = Var(domain=PositiveIntegers)  # the objective function = max distance to be minimized

  #constraints
  #each item should be delivered by only one courier
  def one_item_delivery_rule(model,j):
      return sum(model.x[k,i,j] for k in model.K for i in model.loc if i!=j) == 1
  model.one_item_delivery = Constraint(model.I, rule=one_item_delivery_rule)

  #enter and exit from node, number of times exit a node i = number of times enter a node i
  def enter_exit_rule(model,k,i):
      return sum(model.x[k,i,j] for j in model.loc) == sum(model.x[k,j,i] for j in model.loc)
  model.enter_exit = Constraint(model.K, model.loc, rule=enter_exit_rule)

  #each courier start from depot exactly once
  def leave_depot_rule(model,k):
      return sum(model.x[k,n,j] for j in model.I) == 1
  model.depot_leave = Constraint(model.K, rule=leave_depot_rule)

  #each courier return to depot exactly once
  def return_depot_rule(model,k):
      return sum(model.x[k,i,n] for i in model.I) == 1
  model.depot_return = Constraint(model.K, rule=return_depot_rule)

  #avoid self-loop
  def self_loop_rule(model,k,i):
      return model.x[k,i,i] == 0
  model.self_loop = Constraint(model.K, model.loc, rule=self_loop_rule)

  #capacity constraint
  def capacity_rule(model,k):
      return sum(size[i]*(model.x[k,i,j]) for i in model.I for j in model.loc if i!=j) <= capacity[k]
  model.capacity = Constraint(model.K, rule=capacity_rule)

  #prevent subtours with MTZ constraint, used in TSP problem
  #https://phabe.ch/2021/09/19/tsp-subtour-elimination-by-miller-tucker-zemlin-constraint/
  model.order = ConstraintList()
  for k in model.K:
      for i in model.I:
        for j in model.I:
          if i != j:
            #use Big-M notation(M=2*n)
            model.order.add(model.o[k, j] - model.o[k,i] >= 1 -(1-model.x[k,i,j])*2*n)

  #compute distance
  def total_distance_rule(model,k):
      return model.dist[k] == sum(dist[i][j] * model.x[k, i, j] for i in model.loc for j in model.loc if i != j)
  model.total_distance_constraint = Constraint(model.K, rule=total_distance_rule)

  #constraint the obj to be maximum distance travelled by any courier
  def max_distance_constraint_rule(model, k):
      return model.obj >= model.dist[k]
  model.max_distance_constraint = Constraint(model.K, rule=max_distance_constraint_rule)

  #lowerbound constraint
  def lower_bound_rule(model,i):
      return model.obj >= (dist[n][i] + dist[i][n])
  model.lower_bound = Constraint(model.I, rule=lower_bound_rule)

  #upperbound:  m-1 courier delivers m items, and the last courier deliver n-m+1, take n-m-1 items with the biggest distances
  sorted_distances=sorted([dist[n][i]+dist[i][n] for i in range(n)],reverse=True)
  upper_bound = sum(sorted_distances[:n-m+1])
  def upper_bound_rule(model):
      return model.obj <= upper_bound
  model.upper_bound = Constraint(rule=upper_bound_rule)

  #objective
  model.objective = Objective(expr = model.obj, sense=minimize)
  return model


# %%
#GLPK SOLVER
def solve_with_glpk(model:ConcreteModel):
  start_time = time.time()
  solver = SolverFactory('glpk')
  solver.options['tmlim'] = 300
  results = solver.solve(model,tee=False)
  final_time = time.time() - start_time
  print(results)
  #print(final_time)
  #print(model.obj.value)
  return final_time

# %%
#CBC SOLVER
def solve_with_cbc(model:ConcreteModel):
  start_time = time.time()
  solver = Cbc()
  solver.config.time_limit = 300
  results = solver.solve(model)
  final_time = time.time() - start_time
  print(results)
  #print(final_time)
  #print(model.obj.value)
  return final_time

# %%
#HIGHS SOLVER
def solve_with_highs(model:ConcreteModel):
  start_time = time.time()
  solver = Highs()
  solver.config.time_limit = 300
  results = solver.solve(model)
  final_time = time.time() - start_time
  print(results)
  #print(final_time)
  #print(model.obj.value)
  return final_time

# %%
def extract_solution(model:ConcreteModel, show_edges=False):
  sol = []
  n = len(model.I)
  for k in model.K:
    sub_sol = []
    for j in model.loc:
      #start the route from the depot and deliver the first item
      if value(model.x[k,n,j]) > 0.5:
        first = j
        sub_sol.append(first+1)
        break
      #print the edges traversed by courier k if show is True
      if show_edges:
        for i in model.loc:
          for j in model.loc:
            if value(model.x[k, i, j]) > 0.5:
              print(f"travels from {i} to {j}")
    #route extraction(items delivered by the courier k following the delivery order)
    succ=0
    prec=first
    while(True):
      for i in model.loc:
        if value(model.x[k,prec,i]>0.5):
          if i == n:
            succ=n
          prec=i
          break
      if succ == n:
        break
      sub_sol.append(prec+1)
    #print(sub_sol)
    sol.append(sub_sol)
  print(sol)
  return sol


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

BASE_PATH = os.path.dirname(__file__)

def run_mip_and_save(data_instance_number:int, result_file: str):
  dat_file = BASE_PATH + f"/../Instances/inst{'{:0>2}'.format(data_instance_number)}.dat"
  out = {}

  model = build_model_from_data_file(dat_file)
  time = solve_with_glpk(model)
  sol = extract_solution(model)
  out["glpk"] = sol_to_dict(time, model.obj.value, sol)

  model = build_model_from_data_file(dat_file)
  time = solve_with_cbc(model)
  sol = extract_solution(model)
  out["cbc"] = sol_to_dict(time, model.obj.value, sol)

  model = build_model_from_data_file(dat_file)
  time = solve_with_highs(model)
  sol = extract_solution(model)
  out["highs"] = sol_to_dict(time, model.obj.value, sol)

  save_to_file(out, result_file)




