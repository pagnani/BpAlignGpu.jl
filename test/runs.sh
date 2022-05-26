#!/bin/sh 

inds=(17 21 38 48 50 56 57 60)
for idx0 in ${inds[@]}; 
do 
    echo $idx0;
    julia findGS.jl $idx0 0.7 0.5 >> out_runs; 
    sleep 3; 
done;
