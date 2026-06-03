#!/bin/bash

#Defintion of the problems to execute
PROBLEMS=("ARGLINA" "BARD" "BEALE" "BRKMCC" "BROWNAL" "BROWNBS" "BROWNDEN" "CHNROSNB" "CLIFF" "CUBE" "DECONVU" \
    "DENSCHNA" "DENSCHNB" "DENSCHNC" "DENSCHND" "DENSCHNF" "" "ENGVAL2" "EXTROSNB" \
     "FLETCHCR" "HAIRY" "HEART6LS" "HELIX" "HILBERTA" "HILBERTB" "HIMMELBB" "HIMMELBH" "HUMPS" \
    "JENSMP" "KOWOSB" "LOGHAIRY" "MANCINO" "MARATOSB" "MEXHAT" "PALMER1C" "PALMER2C" "PALMER3C" "PALMER4C" "PALMER5C" \
    "PALMER6C" "PALMER7C" "PALMER8C" "ROSENBR" "SINEVAL" "SISSER" "TOINTQOR" "VARDIM" "WATSON" "YFITU" "BIGGS6" "BOX3" \
    "GAUSSIAN" "GULF" "PENALTY1" "PENALTY2" "TRIGON1" "WOODS")

#PROBLEMS=("MANCINO")

#Available methods
METHODS=("AMG" "SGLS" "QUEUE" "RANDOM")

#Execution of the methods using a for loop in Bash
for prob in "${PROBLEMS[@]}"; do
    echo "=========================================================="
    echo "Running experiments for: $prob"
    echo "=========================================================="
    for conf in "${CONFIGURATIONS[@]}"; do
        echo "  -> Method: $conf"
        julia separte_execution.jl --problem 1-cutest-sif/$prob.SIF --config "$conf" --method BFGS --seed 42 
    done
done

echo "All experiments finished!"