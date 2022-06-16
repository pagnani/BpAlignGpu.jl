#!/bin/sh 

inds=(11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
for idx0 in ${inds[@]}; 
do 
    echo $idx0;
    julia epsilon-coupling.jl $idx0 >> out_runs_epsilon
    sleep 3; 
done;

