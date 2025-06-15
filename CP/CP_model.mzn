include "globals.mzn";

%parameters
int: n;    % Number of items
int: m;    % Number of couriers

set of int: items = 1..n;
set of int: couriers = 1..m;
set of int: nodes = 1..n+1; % Including origin point

array[1..n+1,1..n+1] of int: dist;  % Distance matrix
array[items] of int: size;    % Size of each item
array[couriers] of int: capacity;          % Capacity of each courier

%variable
array[couriers,nodes] of var 1..n+1:succ;

array[1..m] of var int: tot_dist;
var int: max_distance;

%constraints
%each item(exclude the depot node in the iteration) must be assigned only to only one courier
%constraint forall(i in items) (
 %       count(s in succ[..,i])(s != i) == 1 
  % );
constraint forall(i in items) (
        sum(k in couriers)(succ[k,i] != i) == 1 
    );
% load constraint
constraint forall(k in couriers)(
  sum(i in items where succ[k,i] != i)(size[i])  <= capacity[k]
  );

%constraint each courier should enter and exit the depot only once
constraint forall(k in couriers)(
          count(s in succ[k,..])(s = n+1)== 1 /\ succ[k,n+1] != n+1
    );
% constraint each courier route schould be an Hamiltonian sub-tour (or sub-circuits)
constraint forall(k in couriers) (
      subcircuit(succ[k,..])
    );
%objective
constraint forall(k in couriers)(
    tot_dist[k] = sum(i in 1..n+1 where succ[k,i] != i)(dist[i,succ[k,i]])
   );
    
%array[couriers] of int: tot_dist  = forall(k in couriers)(sum(i in 1..n+1 where succ[k,i]!=i)(dist[i,succ[k,i]]));
int: obj_lowerbound = max(i in items)(dist[n+1,i] + dist[i,n+1]);
var int: obj = max(tot_dist[..]);
constraint obj >= obj_lowerbound;

% Search Strategy
%-----------------------------------------------------------------------------%
solve :: int_search(succ, dom_w_deg, indomain_min) 
%-----------COMMENT TO REMOVE LNS and RESTART-----------%
     % :: restart_luby(100)
      :: relax_and_reconstruct(array1d(succ), 85)
%-----------COMMENT TO REMOVE LNS and RESTART-----------%
         minimize obj; 
