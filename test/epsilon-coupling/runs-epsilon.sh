#!/bin/sh 

inds=(51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100)
for idx0 in ${inds[@]}; 
do 
    echo $idx0;
    julia epsilon-coupling.jl $idx0 >> out_runs_epsilon
    sleep 3; 
done;


