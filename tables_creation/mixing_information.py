import numpy as np
import pandas as pd
import argparse

#To modify the paths
from pathlib import Path
import os
import glob

#Declaration of variables were we will load the results.
parser = argparse.ArgumentParser(prog="Appending CSV for problems")

#Parser values
parser.add_argument("-p", "--problems_path", help="Path to the problems results")


"""
# Script intention

This script is intended to joint all the results in the noise robustness experiments. According
to the area_main.jl, the results are saved at PWD/csvs/results/area_robust/PROBLEM; using the
problem name we can read all the results of the methods across all the configurations.
By construction we know that such folder contains three sub-folders, one
for simplest, other for original and another for WithLS. Therefore, 
we access such paths, then we load the cvs of each method, and then we
join all of them into just one CSV file. 

This script generates a folder with three different csvs files for each configuration. With
the following format.

PWD/csvs/results/PROBLEM/Joint_CSV/CONF.csv

Where PROBLEM is the name of the optimization problem, 'Joint_CSV' is a hardcoded path name where the
joined methods will be saved, and CONF is one of three configurations (Simplest, original, WithLS).

## Execution instance
python3 mixing_information.py -p ../csvs/results/area_robust/


# Contact information

Email: jose.quiroz@cimat.mx
Alias: @Qntmth || @Qntmth-uv (Github)
Date: May 18, 2026

"""

def join_results(results: dict, order:list[str] = ["AMG", "Queue", "Random", "SGLS"])->pd.DataFrame:
    """
    ## Definition
    Creates a final dataframe with the all the results of the problem with the following order of rows
        1. Simplest
        2. Original
        3. WithLs
        4. SGLS
    This is nor changeable
    
    ### Inputs:
        #### Required.
            - results:(dict) - A dictionary such as the keys are the available methods and the values
                                 are the dataframes.

    ### Outputs:
        - mixed_df(pd.DataFrame) - A dataframe with all the results of all methods.

    ### Remark.
    We assume that the order of the list in the dataframes are
    [simplest, original, WithLS, SGLS].
    """
    #Assertion that the length of the given order is the same that the available methods.
    assert len(results.keys()) == len(order), f"The length of the results array and the given order does not match. ({len(results.keys)},{len(order)})"

    #Construction of the final dataframe
    final_df:pd.DataFrame = pd.concat([results[meth] for meth in order])
    return final_df

def existing_folders(path: Path)->list[str]:
    """
    ### Definition.
    List the folders in the given path.

    #### Inputs:
        ##### Required.
        - path(Path): Folder directory where to search the current folders.

    #### Outputs:
        - problems_names(list[str]): Name of the folders in the given path.    
    
    ##### Remark.
    In the context of noise robustness, this function is used to list all
    the available problems. With the given list, we will generate the joined 
    dataframes.

    """
    return [f.name for f in path.iterdir() if f.is_dir()]


def read_csvs(path: Path, methods:tuple[str])->list[pd.DataFrame]:
    """
    ## Definition
    Function to load the CSVS files with the results of the given configurations.

    ### Input:
        #### Required:
            - path(Path): Directory where the results of the methods are stored.
            - methods(tuple): In-mutable order in which the CSVS files will be load.

    ### Outputs:
        - dataframes(list[pd.DataFrames]): List where are the CSVS files in pandas Dataframe class.

    #### Remarks.
    This is an special case of the the other defined functions to load the CSVS files
    in others scrips. However, there are a small number of CSVS files to read. That's
    why use this small function.
    """

    #Read the following files in the given path
    csv_files = [os.path.join(path, m.lower()) + ".csv" for m in methods]

    #Convert all the csvs in pandas dataframes
    try:
        dataframes = [pd.read_csv(f) for f in csv_files]
    except:
        raise ValueError(f"Error reading a CSVS file in the path: {path}")
    return dataframes


def main()->None:
    #Args
    args = parser.parse_args()
    path = Path(args.problems_path)
 
    #Configurations
    configs = ["simplest", "original", "addedLS"]    
    methods = ("AMG", "Queue", "Random", "SGLS")

    print("Reading from path:", path)
    existing_problems = existing_folders(path)
    available_paths = [Path(os.path.join(path, p)) for p in existing_problems] 
    #available_paths = [Path(os.path.join(path, "BRKMCC"))]
    
    for p in available_paths:
        #Path where to storage the joined results.
        writing_path = os.path.join(p, "Joined_CSVS")

        #Checking if the Joined_CSVS already exist, if it's not then we created
        if not os.path.exists(writing_path):
            os.mkdir(path=writing_path)
            print(f"Created path to save the joined files at: {writing_path}")
        else:
            print("The 'Joint_CSVS' already exist. No new directory was created.")

        #Get the paths of each configurations of the given problem
        configurations_paths = [os.path.join(p, c) for c in configs]

        #Get the csvs files
        csvs_per_Config = [read_csvs(conf, methods) for conf in configurations_paths]

        #Creation of the dictionaries with their corresponding dataframe    
        dicts = [dict(zip(methods, l)) for l in csvs_per_Config]

        #Create the dataframe with the joined results
        final_df = [join_results(d, methods) for d in dicts]
        
        #Write each of the joined dataframes
        for (df, c) in zip(final_df, configs):
            file_path = os.path.join(writing_path, c+".csv")
            df.to_csv(file_path, index=False)
            print(f"---> The CSV file for {c} configuration was created at: {file_path}")
        
    return 1

if __name__ == "__main__":
    main()
