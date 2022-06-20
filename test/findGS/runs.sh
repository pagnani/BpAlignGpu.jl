#!/bin/sh 

inds=(65 69 70 76 85 96)
for idx0 in ${inds[@]}; 
do 
    echo $idx0;
    julia findGS.jl $idx0 0.7 0.5>> out_runs; 
    sleep 3; 
done;
