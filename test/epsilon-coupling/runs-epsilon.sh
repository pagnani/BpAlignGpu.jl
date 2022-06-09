#!/bin/sh 

inds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
epsilons=(-0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0 -1.2 -1.3 -1.5)
for idx0 in ${inds[@]}; 
do 
    for eps in ${epsilons[@]};
    do
        echo $idx0 $eps;
        julia epsilon-coupling.jl $idx0 $eps 0.2 >> out_runs_epsilon
        sleep 3; 
    done;
done;

