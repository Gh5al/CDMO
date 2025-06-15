int: n;    % Number of items
int: m;    % Number of couriers

set of int: items = 1..n;
set of int: couriers = 1..m;
set of int: loc = 1..n+1; % Including origin point

array[1..n+1,1..n+1] of int: dist;  % Distance matrix
array[items] of int: size;    % Size of each item
array[couriers] of int: capacity;          % Capacity of each courier

array[1..m, 1..n+1, 1..n+1] of var 0..1: x;%x[k][i][j]= 1 states that the courier k move from i to j
%array[1..n] of var couriers: assign;
array[1..m,1..n] of var 1..n: u;

array[1..m] of var int: distance;
var int: max_distance;

%each courier must start at depot
constraint forall(k in couriers)(
    sum(j in 1..n) (x[k,n+1,j]) = 1  % Each courier leaves the depot exactly once
);
%each courier must arrive at depot
constraint forall(k in couriers)(
    sum(i in 1..n) (x[k,i,n+1]) = 1  % Each courier arrive at the depot exactly once
);
%constraint if a courier already passed from that location, another courier shouldn't pass from there
%constraint forall(k in couriers)(
% forall (i in 1..n)(
%  if (assign[i]=k) then forall(q in couriers where q!=k)(forall(j in 1..n)(x[q,i,j]=0))endif) 
%);

%inside cycle avoid going to same location more than once

%constraint sum(j in 1..n,k in 1..m)(x[k,n+1,j]) = sum(j in 1..n,k in 1..m)(x[k,j, n+1]);

%Ensure that Every Node is Entered Once and left by the same veichle
constraint forall(j in 1..n)(
 sum(k in couriers, i in 1..n+1)(x[k,i,j])=1);


constraint forall(k in couriers)(
forall(i in 1..n,j in 1..n where i!=j)(
   % u[k,j] - u[k,i] >= size[j] - capacity[k] * (1-x[k,i,j]))
   %u[k,i] - u[k,j] + (n+1)*x[k,i,j] <= n)
   u[k,i] - u[k,j] +1 <= (n-1)*(1-x[k,i,j]))
);

% avoid counting the value on diagonal
constraint forall(k in couriers)(
    forall(i in 1..n+1)( (x[k,i,i]) = 0)  
);

%each courier can move towards only one another node
%constraint forall(k in couriers)(
 % forall (i in 1..n+1)(
  %  sum(j in 1..n+1)(x[k,i,j]) <=1));


%assign fill should be consistent with x
%constraint forall(k in couriers)(
 % forall(i in 1..n)(
 %   (assign[i] = k) <-> (sum(j in 1..n) (x[k,i,j]) > 0))
%);


% Capacity constraint
%constraint forall(k in couriers) (
 %   sum([size[j] * (assign[j] = k) | j in items]) <= capacity[k]  
%);

constraint forall(k in couriers)(
  sum(i in 1..n+1, j in 1..n)(size[j]*x[k,i,j])<=capacity[k]);

%if a courier goes to a location then it must leave
constraint forall(k in couriers)(
forall(i in 1..n+1)(
    sum(j in 1..n+1) (x[k,i,j]) = sum(j in 1..n+1) (x[k,j,i]))
);

int: obj_lowerbound = max(i in 1..n)(dist[n+1,i] + dist[i,n+1]);

constraint forall(k in 1..m)(
    %distance[k] = sum(i in 1..n+1, j in 1..n+1)( x[k,i,j] * dist[i, j])
    distance[k] = sum(i in 1..n+1)(sum (j in 1..n+1)(x[k,i,j] * dist[i,j]))

);


var int: max_dist = max([distance[k] | k in 1..m]);
%solve :: int_search([x[k, i, j] | k in 1..m, i in 0..n, j in 0..n], input_order, indomain_random, complete)

constraint max_dist >= obj_lowerbound;
%solve :: int_search(x, dom_w_deg, indomain_min) 
 %     :: restart_luby(100)
  %    :: relax_and_reconstruct(array1d(x), 85)
   %   minimize max_distance;
solve minimize max_dist;