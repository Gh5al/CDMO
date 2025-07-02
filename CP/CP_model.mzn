include "globals.mzn";

%parameters
int: n;    % Number of items
int: m;    % Number of couriers

set of int: items = 1..n;
set of int: couriers = 1..m;
set of int: locs = 1..n+1; % Locations (including depot)

array[locs,locs] of int: dist;  % Distance matrix
array[items] of int: size;    % Size of each item
array[couriers] of int: capacity; % Capacity of each courier

%variable
array[couriers,locs] of var locs:succ;
array[couriers] of var int: tot_dist;
var int: obj;

%constraints
%each item(exclude the depot node in the iteration) must be assigned exactly to one courier
constraint forall(i in items) (
        sum(k in couriers)(succ[k,i] != i) == 1 
    );
% max capacity constraint
constraint forall(k in couriers)(
  sum(i in items where succ[k,i] != i)(size[i])  <= capacity[k]
  );

%constraint each courier should enter and exit the depot only once
constraint forall(k in couriers)(
          count(s in succ[k,..])(s = n+1)== 1 /\ succ[k,n+1] != n+1 
    );
% constraint each courier route should be an Hamiltonian sub-tour(prevent unconnected routes)
constraint forall(k in couriers) (
      subcircuit(succ[k,..])
    );
% ----- IMPLIED -----    
%both are covered by subcircuit (remove)
%constraint the couriers leave the depot for different locations (implied)(computationally expensive)
%constraint alldifferent(succ[..,n+1]);

%implied constraint, for each courier the value of successor should be all different(computationally expensive and not neccesary)
%constraint forall(k in couriers)(
  %alldifferent(succ[k,..]));
  
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

solve minimize obj;

output  [show(succ[k,i]) ++ if i == n+1 then "\n" else " " endif | k in couriers, i in locs] ++
        [show(obj) ++ "\n"]
