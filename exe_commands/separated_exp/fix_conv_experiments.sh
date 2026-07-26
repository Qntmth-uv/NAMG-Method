#!/bin/bash

#Defintion of the problems to execute
PROBLEMS=("WATSON")

#PROBLEMS=("MANCINO")

#Available methods

CONFIGURATIONS=("simplest" "original" "addedLS")

#Execution of the methods using a for loop in Bash
for prob in "${PROBLEMS[@]}"; do
    echo "=========================================================="
    echo "Running experiments for: $prob"
    echo "=========================================================="
    for conf in "${CONFIGURATIONS[@]}"; do
        echo "  -> Method: $conf"
        julia --project=venv_NAMGM separte_execution.jl --problem 1-cutest-sif/$prob.SIF --config "$conf" --method BFGS --seed 42 
    done
done

echo "All experiments finished!"
