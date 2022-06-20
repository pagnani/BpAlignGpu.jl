#!/bin/sh 

inds=(26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
for idx0 in ${inds[@]}; 
do 
    echo $idx0;
    julia epsilon-coupling.jl $idx0 >> out_runs_epsilon
    sleep 3; 
done;

