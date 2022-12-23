#!/bin/sh 

for idx0 in $(cat indices_seedPF00397.txt); 
do 
    echo $idx0;
    julia findGS.jl $idx0 0.7 0.9 >> out_runs; 
    sleep 3;     
done

--------------------------------------------------------
inds=(48 60 96 24 65)
for idx0 in ${inds[@]}; 
do 
    echo $idx0;
    julia findGS.jl $idx0 0.7 0.9 >> out_runs; 
    sleep 3; 
done;
