import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

"""
#Script intention
This script solves the problem of updating the results of the main.jl . We make this
trough the insertion of the results and we save them on a new csvs/<NEW_FOLDER>/<CONFIGS>/PROBLEMS.
This to do not overwrite the first results. 
"""
#Constants variables
CONFIGS = ("simplest", "original", "addedLS")
CSVS_path = "csvs/results/"
VERSION = "1_0_1"


def get_ListOfDF(directory:str, change_order:bool = False):
    # Read all the results tables in the directory
    csv_files = glob.glob(directory)

    # Sort the tables by name
    csv_files = sorted(csv_files)

    if change_order:
        temp = csv_files[0] #addedLS
        #Interchange of Simplest with addedLS
        csv_files[0] = csv_files[-1] 
        csv_files[-1] = temp 

    # Read all the csv files from the directory, and save we save them in a list
    dataframes = [pd.read_csv(f) for f in csv_files]
    names_frames = [Path(csv).stem.split('_')[0] for csv in csv_files]
    
    return dataframes, names_frames

def surgery_df(insertion_method_number, original_dfs: list[list[pd.DataFrame]], method_dfs: list[list[pd.DataFrame]]):
    #Assertion of each dfs
    assert len(original_dfs) == len(method_dfs), "The lists does not have the same size"

    #Assertion that both experiments have the same order.
    for i in range(len(original_dfs)):
        assert len(original_dfs[i]) == len(method_dfs[i]), "The order and number of problems are not the same"

    #Now we know that the list have the exact same problems in the correct order, me make the surgery
    for i in range(len(original_dfs)):  #Over configurations
        for j in range(len(original_dfs[0])): #Over problems
            original_dfs[i][j].iloc[insertion_method_number, :] = method_dfs[i][j].to_numpy()

    return original_dfs

def main():

    method = "bfgs"

    #Legacy values 
    folders_orginal = [os.path.join(CSVS_path, c, "*.csv") for c in CONFIGS]
    dfs_originals = [get_ListOfDF(f, False)[0] for f in folders_orginal]
    problems_names_original = [get_ListOfDF(f, False)[1] for f in folders_orginal]

    #New results per method
    folders_method = [os.path.join(CSVS_path, method, c, "*.csv") for c in CONFIGS]
    dfs_method = [get_ListOfDF(f, False)[0] for f in folders_method]
    problems_names_methods = [get_ListOfDF(f, False)[1] for f in folders_method]

    """Assertion of the problems, that the match in each configuration"""
    assert len(problems_names_original) == len(problems_names_methods), "The number of configurations are not the same"
    for i in range(len(problems_names_methods)):
        problems_o = problems_names_original[i]
        problems_m = problems_names_methods[i]
        for o, m in zip(problems_o, problems_m):
            assert o==m, f"The List of problems are not the same at {CONFIGS[i]} | Original: {o} | Method: {m}|"

    #Modified results
    modified_dfs = surgery_df(4, dfs_originals, dfs_method)

    #Creation a new folder for the modified results
    new_folder_with_dfs = CSVS_path+f"UpdatedResults_{VERSION}"
    os.makedirs(new_folder_with_dfs, exist_ok=True)

    #Now we create the folders for the different configs
    for c in CONFIGS:
        os.makedirs(os.path.join(new_folder_with_dfs, c), exist_ok=True)

    #We save in the new folder
    for i in range(len(modified_dfs)):
        conf_list = modified_dfs[i]
        for problem, df in zip(problems_names_original[0], conf_list):
            path_to_save_problem = os.path.join(new_folder_with_dfs, CONFIGS[i], f"{problem}.csv")
            df.to_csv(path_to_save_problem)
            print(f"Updated the problem {CONFIGS[i]}/{problem} at :{path_to_save_problem}")
    return

if __name__ == "__main__":
    main()