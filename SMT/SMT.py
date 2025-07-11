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
from z3 import Solver, sat, Int, Bool, Sum, If, Or, And, Not, Distinct, Implies
import numpy as np
import time
#from multiprocessing import Process, Queue
import json
import os
import math
import re
import signal


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



def sol_to_dict(t:int, obj:str, sol:list, solved:bool):
  if t<300 and obj != -1 and solved:
    optimal = True
    time = math.floor(t)
  elif t >= 300:
    print("Time limit reached:", t)
    optimal = False
    time = 300
  else:
    optimal = False
    time = math.floor(t)

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

def search_solution(solver:Solver, objective:Int, m:int, n:int, lower_bound:int, pos:list):
  print("Searching solution...")
  search_start_time = time.time()

  #--------------------------------- SEARCHING ---------------------------------
  curr_obj = -1
  
  #For some instances the solver doesn't abort the searching process even after timeout
  signal.alarm(5*60 + 1)
  while True:
    timeoutS = search_start_time + 5*60 - time.time()
    if timeoutS <= 0:
      solved = False
      print('Timeout already reached')
      break
    
    solver.set(timeout=int(timeoutS*1000)) # https://microsoft.github.io/z3guide/programming/Parameters/#global-parameters
    try:
      if solver.check() != sat:
        print("No more solutions available")
        break
    except Exception as e:
      print("Search failed", e)
      solved = False
      break
    
    model = solver.model()
    curr_obj = model.evaluate(objective).as_long()
    print("Current objective:", curr_obj, "; remaining seconds:", timeoutS)
    #routes = extract_routes(model,m,n,d_var,succ=succ)
    if curr_obj <= lower_bound:
      solved = True
      break
    if time.time() - search_start_time >= 5*60:
      solved = False
      print('Timeout reached')
      break
    if solver.check() != sat:
      solved = False
      print('Failed to solve')
      break
    solved = True
    #try to improve the objective adding an upperbound with the current objective
    solver.add(objective < curr_obj)
  #--------------------------------- BUILD_SOL ---------------------------------

  if(curr_obj == -1):
    print("No solution found")
    solved = False
    sol = []
  else:
    print("Using curr_obj for solution:", curr_obj)
    routes = extract_routes(model,m,n,pos)
    #extract the items assigned to each courier
    sol = extract_sol(routes,n)
  #run the searching using the Process and Queue library to abort execution after reaching timeout
  #in this approach the routes are extracted from position variable
  #info = run_with_timeout(m,n,solver, pos, objective,lower_bound, timeout)
  #print(info)
  signal.alarm(0)

  print(f"\n\nFinal objective: {curr_obj}")
  searching_time = time.time() - search_start_time
  final_time = searching_time
  print(f"Finished in: {final_time:3.2f} secs\n")
  return final_time,curr_obj, sol, solved

def throw_exception_on_sigtimer(signum, frame):
  """
  https://stackoverflow.com/questions/492519/timeout-on-a-function-call
  https://docs.python.org/3/library/signal.html#signal.alarm
  """
  raise Exception("Forcefully interrupting the research after 5 minutes")



# %% [markdown]
# ## BOOLEAN VARIABLE + POSITION VARIABLE

# %%
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

  #number of assigned items to each courier
  num_assigned = [Int(f"num_assigned_{k}") for k in couriers]

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
    solver.add(Or([A[k][i] for i in items]))

  #capacity constraints
  for k in couriers:
    solver.add(Sum([If(A[k][i],size[i],0) for i in items]) <= capacity[k])

  #for each courier all the position assigned should be different
  for k in couriers:
    solver.add(Distinct([pos[k][i] for i in items]))

  for k in couriers:
    solver.add(num_assigned[k] == Sum([If(A[k][i],1,0) for i in items]))

    #prevent unconnected routes between delivered items by a courier k
    for i in items:
      #if an item is assigned to a courier k, then the position should have a value between 1 and num_assigned items to the courier k
      solver.add(A[k][i] == (pos[k][i] >= 1))
      solver.add(pos[k][i] <= num_assigned[k])
      solver.add(Not(A[k][i]) == (pos[k][i] == -i))

    #Distance calculation
    #distance from depot to the first delivery
    depot_to_first= Sum([If(And(A[k][i], pos[k][i] == 1),dist[n][i],0) for i in items])

    # if an item i and j are delivered, and their position only differs of 1, take all the pairs and then sum all the distances
    between_distance = Sum(
      [If(
        A[k][i],
        Sum([
          If(And(A[k][j],pos[k][j] == pos[k][i]+1), dist[i][j], 0)
          for j in items
          if j!=i
        ]),
        0
      ) for i in items]
    )

    #distance from last delivery to depot
    last_to_depot = Sum([If(And(A[k][i], pos[k][i] == num_assigned[k]),dist[i][n],0) for i in items])

    #total distance
    solver.add(distance[k] == depot_to_first + between_distance + last_to_depot)

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

  return search_solution(solver, objective, m, n, lower_bound, pos)


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
  
  return search_solution(solver, objective, m, n, lower_bound, pos)

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

  signal.signal(signal.SIGALRM, throw_exception_on_sigtimer)

  print("Running boolean variable model...")
  t, obj, sol, solved = run_boolean_model(dat_file)
  out["boolean"] = sol_to_dict(t, obj, sol, solved)

  print("Running assign int variable model...")
  t, obj, sol, solved = run_int_assign_model(dat_file)
  out["assign_int"] = sol_to_dict(t, obj, sol, solved)

  save_to_file(out, result_file)


