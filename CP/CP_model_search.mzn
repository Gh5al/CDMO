include "globals.mzn";

%PARAMETERS
int: n;    % Number of items
int: m;    % Number of couriers

set of int: items = 1..n;
set of int: couriers = 1..m;
set of int: locs = 1..n+1; % Locations (including depot)

array[locs,locs] of int: dist;  % Distance matrix
array[items] of int: size;    % Size of each item
array[couriers] of int: capacity; % Capacity of each courier

%VARIABLES
array[couriers,locs] of var locs:succ;
array[couriers] of var int: tot_dist;
var int: obj;

%CONSTRAINTS

%each item(exclude the depot node in the iteration) must be assigned exactly to one courier
constraint forall(i in items) (
      sum(k in couriers)(succ[k,i] != i) == 1 
    );
%capacity constraint
constraint forall(k in couriers)(
      sum(i in items where succ[k,i] != i)(size[i])  <= capacity[k]
  );

%constraint each courier should leave the depot 
constraint forall(k in couriers)(
      succ[k,n+1] != n+1 
    );
%constraint each courier's route should be an Hamiltonian sub-tour(prevent unconnected routes)
constraint forall(k in couriers) (
      subcircuit(succ[k,..])
    );
% ----- IMPLIED -----    
%constraint each courier should enter the depot only once
constraint forall(k in couriers)(
      count(s in succ[k,..])(s = n+1)== 1);
%each courier should carry at least one item, this is covered by the leave depot constraint
constraint forall(k in couriers)(
      sum(i in items)(succ[k,i]!=i) >= 1);
% ----- IMPLIED -----    
    
%compute tot_distance travelled by each courier
constraint forall(k in couriers)(
      tot_dist[k] = sum(i in locs where succ[k,i] != i)(dist[i,succ[k,i]])
   );

%objective function, the maximum distance travelled by any courier
obj = max(tot_dist[..]);

%define the lowerbound of the objective as the biggest distance to deliver an item from depot and return to depot
int: obj_lowerbound = max(i in items)(dist[n+1,i] + dist[i,n+1]);
constraint obj >= obj_lowerbound;

%uppebound: m–1 couriers each deliver one item, the remaining courier delivers the n–m+1 items.
%consider the worst case scenario: take the n–m+1 items with the largest distances
array[items] of int: sorted_item_dist = sort([dist[n+1, i] + dist[i, n+1] | i in items]); 
int: upper_bound = sum(i in n-m+1..n)(sorted_item_dist[i]);
constraint obj <= upper_bound;

%SEARCHING
%Search strategy Luby+LNS, with indomain_median
solve :: int_search(succ, dom_w_deg, indomain_median) 
      :: restart_luby(70)
      :: relax_and_reconstruct(array1d(succ), 70)
      minimize obj; 
%solve minimize obj;

output  [show(succ[k,i]) ++ if i == n+1 then "\n" else " " endif | k in couriers, i in locs] ++
        [show(obj) ++ "\n"]
