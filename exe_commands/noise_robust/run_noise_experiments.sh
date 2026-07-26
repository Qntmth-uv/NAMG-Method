#!/bin/bash

#Defintion of the problems to execute
PROBLEMS=("ARGLINA" "BRKMCC" "BROWNBS" "CLIFF" "DENSCHNA" "DENSCHNC" "DENSCHNF" "GAUSSIAN" "HILBERTA" "HILBERTB" "HIMMELBB" \
          "HIMMELBH" "JENSMP" "MARATOSB" "PENALTY1" "ROSENBR" "SINEVAL" "SISSER" "VARDIM")

#Available methods
METHODS=("AMG" "SGLS" "QUEUE" "RANDOM")

#Execution of the methods using a for loop in Bash
for prob in "${PROBLEMS[@]}"; do
    echo "=========================================================="
    echo "Running experiments for: $prob"
    echo "=========================================================="

    #We create the entry to save the results of the execution
    mkdir -p "exe_histo/noise_robust/$prob/"{original,simplest,addedLS}

    #We iterate over the available methods
    for method in "${METHODS[@]}"; do
        
        # Convert the uppercase method name (e.g., "AMG") to lowercase for the text file ("amg.txt")
        # Using 'tr' is highly portable and works perfectly in both Bash and Zsh.
        method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
        file_name="${method_lower}.txt"

        echo "  -> Method: $method"

        # --- Original Configuration ---
        julia --project=venv_NAMGM area_main.jl --method "$method" --problem "1-cutest-sif/$prob.SIF" --subdirectory original --show_info --seed 42 --modifierH eigen --saveinfo | tee "exe_histo/noise_robust/$prob/original/$file_name"

        # --- Simplest Configuration ---
        julia --project=venv_NAMGM area_main.jl --method "$method" --problem "1-cutest-sif/$prob.SIF" --subdirectory simplest --show_info --seed 42 --modifierH none --saveinfo | tee "exe_histo/noise_robust/$prob/simplest/$file_name"

        # --- AddedLS Configuration ---
        julia --project=venv_NAMGM area_main.jl --method "$method" --problem "1-cutest-sif/$prob.SIF" --subdirectory addedLS --show_info --seed 42 --modifierH eigen --useLS --saveinfo | tee "exe_histo/noise_robust/$prob/addedLS/$file_name"
    done
done

echo "All experiments finished!"