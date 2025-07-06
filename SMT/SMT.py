# %% [markdown]
# <a href="https://colab.research.google.com/github/Gh5al/CDMO/blob/main/SMT/SMT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# We tried 3 methods:
# 
# - boolean assignment variable + postion variable,
# - int assign variable (assign a courier to each item) + position variable
# - successor approach with position variable (not working with some instances)
# .<br>
# 
# For all the approaches it's fundamentel to use the lower bound constraint as it helps to reduce the search space significantly and provide a solution to instances.

# %%
from z3 import Solver, sat, Int, Bool, Sum, If, And, Not, Distinct, Implies
import numpy as np
import time
#from multiprocessing import Process, Queue
import json
import os
import math
import re


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



def sol_to_dict(t:int, obj:str, sol:list):
  if t<300 and obj != -1:
    optimal = True
    time = math.floor(t)
  elif obj == -1:
    optimal = False
    if t >= 300:
      time = 300
    else:
      time = math.floor(t)
  else:
    optimal = False
    time = 300

  # Create JSON structure
  return {
        "time": time,
        "optimal": optimal,
        "obj": int(obj),
        "sol": sol
    }


# %%
def extract_routes(model,m,n,d_var,succ=False):
  #extract route
  couriers = list(range(m))
  items = list(range(n))
  locs = list(range(n+1))
  if succ:
    iter = locs
  else: iter = items
  routes = []
  for k in couriers:
    routes.append([model.evaluate(d_var[k][i]).as_long() for i in iter])
  #print(routes)
  return routes

def extract_sol(routes,n,succ=False):
  sol = []
  items = list(range(n))
  #extract solution for successor approach
  if succ:
    for x in routes:
      sub_sol = []
      prev = x[n]
      while(prev != n):
        for i in items:
          if i == prev:
            sub_sol.append(i+1)
            prev = x[i]
            break
      sol.append(sub_sol)
  else:
    #for routes from position variable
    for x in routes:
      sub_sol = []
      #count the number of items delivered by a courier
      count = sum([1 for p in x if p>0])
      #print(count)
      prev = 1
      while(prev != count + 1):
        for i in items:
          if x[i] == prev:
            sub_sol.append(i+1)
            prev = prev+1
      sol.append(sub_sol)
      #print(sub_sol)
  return sol


# %% [markdown]
# ## BOOLEAN VARIABLE + POSITION VARIABLE

#bool var + pos var
def run_boolean_model(filename):

  #read data from instance file
  m,n,capacity,size,dist = read_data(filename)
  print(f"num_couriers:{m}, num_items: {n}")
  items = list(range(n))
  locs = list(range(n + 1))
  couriers = list(range(m))
  solver = Solver()
  start_time = time.time()

  # ----------------------------- VARIABLES ------------------------------------

  # A[k][i] = 1 if courier k delivers item i
  A = [[Bool(f"A_{k}_{i}") for i in items] for k in couriers]

  # pos[k][i]: order index of node i in the route of courier k
  pos = [[Int(f"pos_{k}_{i}") for i in items] for k in couriers]

  # distance[k]: total distance of courier k
  distance = [Int(f"distance_{k}") for k in couriers]

  #objective function: minimize the maximum distance travelled by any courier
  objective = Int("objective")

#--------------------------------- CONSTRAINTS ---------------------------------

  #each item should be delivered by one courier
  for i in items:
      solver.add(Sum([If(A[k][i],1,0) for k in couriers]) == 1)

  #each courier should deliver at least one item
  for k in couriers:
    solver.add(Sum([If(A[k][i],1,0) for i in items]) >= 1)

  #capacity constraints
  for k in couriers:
    solver.add(Sum([If(A[k][i],size[i],0) for i in items]) <= capacity[k])

  #for each courier all the position assigned should be different
  for k in couriers:
    solver.add(Distinct([pos[k][i] for i in items]))

  #prevent unconnected routes between delivered items by a courier k
  for k in couriers:
    #num_assigned items to each courier
    num_assigned = Sum([If(A[k][i],1,0) for i in items])
    for i in range(n):
      #if an item is assigned to a courier k, then the position should have a value between 1 and num_assigned items to the courier k
      solver.add(Implies(A[k][i],And(pos[k][i]>=1,pos[k][i]<=num_assigned)))
      solver.add(Implies(Not(A[k][i]),pos[k][i]<0))

  #Distance calculation
  for k in couriers:
    #num_assigned items to each courier
    num_assigned = Sum([If(A[k][i],1,0) for i in items])

    #distance from depot to the first delivery
    depot_to_first= Sum([If(And(A[k][i], pos[k][i] == 1),dist[n][i],0) for i in items])

    #create 2 for with i,j, if an item i and j are delivered, and their position only differs of 1, take all the pairs and then sum all the distances
    betweem_distance = Sum([Sum([ If(And(A[k][i],A[k][j],pos[k][j] == pos[k][i]+1),dist[i][j],0) for j in items if j!=i]) for i in items])

    #distance from last delivery to depot
    last_to_depot = Sum([If(And(A[k][i], pos[k][i] == num_assigned),dist[i][n],0) for i in items])

    #total distance
    solver.add(distance[k] == depot_to_first + betweem_distance + last_to_depot)

  #constraint the obj to be the biggest distance travelled by any courier
  for k in couriers:
      solver.add(distance[k] <= objective)

  #objective lowerbound
  lower_bound = max([dist[n][i] + dist[i][n] for i in items])
  solver.add(objective>=lower_bound)
  print(f"lower_bound: {lower_bound}")

  #sum all the distances depot-items and items-depot
  #upper_bound = sum([dist[n][i] for i in range(n)]) + sum([dist[i][n] for i in range(n)])

  #m–1 couriers each deliver one item, the remaining courier delivers the n–m+1 items. Worst case: take the n–m+1 items with the largest distances
  sorted_distances=sorted([dist[n][i]+dist[i][n] for i in range(n)],reverse=True)
  upper_bound = sum(sorted_distances[:n-m+1])
  print(f"upper_bound: {upper_bound}")
  solver.add(objective<=upper_bound)

  encoding_time = time.time() - start_time
  print(f"encoding_time: {encoding_time:3.2f} secs\n")
  # Set timeout to 5 minutes(300 secs)(include also the encoding time as for large instances it can be remarkable?)
  timeout = 300
  solver.set("timeout", int(timeout*1000))
  search_start_time = time.time()

  #--------------------------------- SEARCHING ---------------------------------
  curr_obj = -1
  # Try to get intermediate results
  if solver.check() != sat:
    print('Failed to solve')
    return time.time() - search_start_time, -1, []
  #For some instances the solver doesn't abort the searching process even after timeout
  while solver.check() == sat:
    model = solver.model()
    curr_obj = model.evaluate(objective).as_long()
    print(f"current_obj_value: {curr_obj}\n")
    #routes = extract_routes(model,m,n,d_var,succ=succ)
    if curr_obj <= lower_bound:
      break
    if time.time() - search_start_time >= 300:
      break
    if solver.check() != sat:
      print('Failed to solve')
      break
    #try to improve the objective adding an upperbound with the current objective
    solver.add(objective < curr_obj)
  #--------------------------------- BUILD_SOL ---------------------------------

  if(curr_obj == -1):
    print("No solution found")
    sol = []
  else:
    routes = extract_routes(model,m,n,pos)
    #extract the items assigned to each courier
    sol = extract_sol(routes,n)
  #run the searching using the Process and Queue library to abort execution after reaching timeout
  #in this approach the routes are extracted from position variable
  #info = run_with_timeout(m,n,solver, pos, objective,lower_bound, timeout)
  #print(info)

  print(f"\n\nFinal objective: {curr_obj}")
  searching_time = time.time() - search_start_time
  final_time = searching_time
  print(f"Finished in: {final_time:3.2f} secs\n")
  return final_time,curr_obj, sol


# %% [markdown]
# # ASSIGN INT VARIABLE + POSITION VARIABLE

# %%
def run_int_assign_model(filename):
  #read from instance file
  m,n,capacity,size,dist = read_data(filename)
  print(f"num_couriers:{m}, num_items: {n}")
  items = list(range(n))
  locs = list(range(n + 1))
  couriers = list(range(m))
  solver = Solver()
  start_time = time.time()

  # ----------------------------- VARIABLES ------------------------------------

  # assign[i] = k if courier k delivers item i
  assign = [Int(f"assign_{i}") for i in items]

  # pos[k][i]: order index of node i in the route of courier k
  pos = [[Int(f"pos_{k}_{i}") for i in items] for k in couriers]

  # distance[k]: total distance of courier k
  distance = [Int(f"distance_{k}") for k in couriers]

  #objective function: minimize the maximum distance travelled by any courier
  objective = Int("objective")

#--------------------------------- CONSTRAINTS ---------------------------------

  #each item should be delivered by a courier, bound the assign variable
  for i in items:
    solver.add(And(assign[i] >= 0, assign[i] <= m-1))

  #each courier should deliver at least one item
  for k in couriers:
    solver.add(Sum([If(assign[i]==k,1,0) for i in items]) >= 1)

  #capacity constraint
  for k in couriers:
    solver.add(Sum([If(assign[i]==k,size[i],0) for i in items]) <= capacity[k])

  #for each courier all the position assigned should be different
  for k in couriers:
    solver.add(Distinct([pos[k][i] for i in items]))

  #prevent unconnected routes between delivered items by a courier k
  for k in couriers:
    #num_assigned items to each courier
    num_assigned = Sum([If(assign[i]==k,1,0) for i in items])
    for i in items:
      #if an item is assigned to a courier k, then the position should have a value between 1 and num_assigned items to the courier k, other < 0
      solver.add(Implies(assign[i]==k, And(pos[k][i]>=1, pos[k][i]<=num_assigned)))
      solver.add(Implies(Not(assign[i]==k),pos[k][i]<0))

  #Distance calculation
  for k in couriers:
    #num_assigned items to each courier
    num_assigned = Sum([If(assign[i]==k,1,0) for i in items])
    #distance from depot to the first delivery
    depot_to_first= Sum([If(And(assign[i]==k,pos[k][i] == 1),dist[n][i],0) for i in items])
    #create 2 for with i,j, if an item i and j is delivered, and their position only differs of 1, take all the pairs and then sum all the distances
    betweem_distance = Sum(
        [Sum([If(And(assign[i]==k,assign[j]==k,pos[k][j] == pos[k][i]+1),dist[i][j],0) for j in items if j!=i]) for i in items])
    #distance from last delivery to depot
    last_to_depot = Sum([If(And(assign[i]==k,pos[k][i] == num_assigned) ,dist[i][n],0) for i in items])
    #total distance
    solver.add(distance[k] == depot_to_first + betweem_distance + last_to_depot)

  #constraint the obj to be the biggest distance travelled by any courier
  for k in couriers:
      solver.add(distance[k] <= objective)

  #compute objective lowerbound
  lower_bound = max([dist[n][i] + dist[i][n] for i in items])
  solver.add(objective>=lower_bound)
  print(f"lower_bound: {lower_bound}")
  #sum all the distances depot-items and items-depot
  #upper_bound = sum([dist[n][i] for i in range(n)]) + sum([dist[i][n] for i in range(n)])

  #m–1 couriers each deliver one item, the remaining courier delivers the n–m+1 items. Worst case: take the n–m+1 items with the largest distances
  sorted_distances=sorted([dist[n][i]+dist[i][n] for i in range(n)],reverse=True)
  upper_bound = sum(sorted_distances[:n-m+1])
  print(f"upper_bound: {upper_bound}")
  solver.add(objective<=upper_bound)


  encoding_time = time.time() - start_time
  print(f"encoding_time: {encoding_time:3.2f} secs\n")
  # Set timeout to 5 minutes(300 secs)(include also the encoding time as for large instances it can be remarkable)
  timeout = 300
  solver.set("timeout", int(timeout*1000))
  search_start_time = time.time()

  #--------------------------------- SEARCHING ---------------------------------

  curr_obj = -1
  # Try to get intermediate results
  if solver.check() != sat:
    print('Failed to solve')
    return time.time() - search_start_time, -1, []
  #For some instances the solver doesn't abort the searching process even after timeout
  while solver.check() == sat:
    model = solver.model()
    curr_obj = model.evaluate(objective).as_long()
    print(f"current_obj_value: {curr_obj}\n")
    #routes = extract_routes(model,m,n,d_var,succ=succ)
    if curr_obj <= lower_bound:
      break
    if time.time() - search_start_time >= 300:
      break
    if solver.check() != sat:
      print('Failed to solve')
      break
    #try to improve the objective adding an upperbound with the current objective
    solver.add(objective < curr_obj)
  #--------------------------------- BUILD_SOL ---------------------------------
  if(curr_obj == -1):
    print("No solution found")
    sol = []
  else:
    routes = extract_routes(model,m,n,pos)
    #extract the items assigned to each courier
    sol = extract_sol(routes,n)
  #run the searching using the Process and Queue library to abort execution after reaching timeout
  #in this approach the routes are extracted from position variable
  #info = run_with_timeout(m,n,solver, pos, objective,lower_bound, timeout)
  #print(info)

  print(f"\n\nFinal objective: {curr_obj}")
  searching_time = time.time() - search_start_time
  final_time = searching_time
  print(f"Finished in: {final_time:3.2f} secs\n")
  return final_time,curr_obj, sol

# %% [markdown]
# # RUN MODEL

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

def run_smt_and_save(data_instance_number:int, result_file: str):
  dat_file = BASE_PATH + f"/../Instances/inst{'{:0>2}'.format(data_instance_number)}.dat"
  out = {}

  print("Running boolean variable model...")
  t, obj, sol = run_boolean_model(dat_file)
  out["boolean"] = sol_to_dict(t, obj, sol)

  print("Running assign int variable model...")
  t, obj, sol = run_int_assign_model(dat_file)
  out["assign_int"] = sol_to_dict(t, obj, sol)

  save_to_file(out, result_file)


