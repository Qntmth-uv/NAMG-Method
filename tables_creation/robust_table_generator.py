import numpy as np
import pandas as pd
import glob 
import math
import argparse
from pathlib import Path

import table_utils as TU


"""
Usage commands 

python3 robust_table_generator.py -o ../csvs/results/robust/original/BOX3/ -w ../csvs/results/robust/addedLS/BOX3/ -s ../csvs/results/robust/simplest/BOX3/ > tab.tex

julia main.jl --problem 1-cutest-sif/GULF.SIF --DEBUG --modifierH none --subdirectory DEBUG-Robust --factorX0 10.0 --DM 
"""


#Parser variables
parser = argparse.ArgumentParser(description="Path to the directories were to search")

#Variables (paths of the factors results) 
parser.add_argument("-s", "--simplest", help="Path to the simplest folder for the problem")
parser.add_argument("-o", "--original", help="Path to the original folder for the problem")
parser.add_argument("-w", "--withLs", help="Path to the addedLS folder for the problem")
parser.add_argument("-v", "--verbose", help="Show information about the read files", action='store_false' )
#parser.add_argument("-d", "--decimals", "Number of decimals in the execution time (could be useful due to small execution times, default=3)", default=3)

def get_ListOfDF(directory:str, verbose:int = 0):
    # Read all the results tables in the directory
    csv_files = glob.glob(directory)

    # Sort the tables by name
    csv_files = sorted(csv_files)

    # Read all the csv files from the directory, and save we save them in a list
    dataframes = [pd.read_csv(f) for f in csv_files]

    #Assert the correct read of the CSV files.
    assert len(dataframes)!=0, f"There was an error trying to read the files in the files directory {directory}"

    #Get the names of the read csv files    
    names_frames_complete = [Path(csv).stem.split('.csv')[0] for csv in csv_files] 
    problem_name = [Path(csv).stem.split('_')[0] for csv in csv_files][0]

    #We print the name of the read tables if it's required
    if verbose in [-1, 0]:  print(names_frames_complete)
    if verbose in [0,1]: print("Read path: ", directory)

    # We print the tables read
    if verbose == 0:
        for i in range(0, len(csv_files)):
            print(f"|{i}| The table has been read it from the directory: {csv_files[i]}")
        print("-"*80)
    if verbose in [-1,0,1]:
        print(f"{len(dataframes)} tables has been read.")

    return dataframes, names_frames_complete , problem_name

def create_matrix_results(dataframes_list: list[pd.DataFrame])->np.ndarray:
    """
    # Definition
    This function create an `np.ndarray` element class. Each element in the dataframe
    has the results from the the three methods, whose have the following order:

    1. NAMGM/AMG
    2. NAMGM/Queue
    3. NAMGM/Random

    And the order of the columns is 

    1. Convergence 
    2. Number of iterations taken
    3. Execution time
    4. Last gradient norm of the sequence
    5. Number of divergences

    Observe that we are not longer measuring the 'iterations per second'. 
    
    ## Inputs
    The big difference is that the creation of each matrix is based in one problem and we not have
    to modify the order of the variables (this is the big difference btw `robust_tables` and `creation_tables`).
    
    - dataframes_list: list[pd.dataframes] - The list of dataframes of the different factors for the problem

    ## Output

    - matrix_results: np.ndarray - Numpy array with the results for the Configuration/Problem

    """
    # We have 5 methods and 5 variables, then in total we have N \times 5 elements in the dataframe
    result_matrix = np.zeros((len(dataframes_list), 15)) #(Number of factors, Variables (5 variables, but 3 methods))

    #Picking of the information of each method (picking in rows)
    for i in range(0, len(dataframes_list)):
        results_variables_per_factor: np.ndarray = dataframes_list[i].to_numpy().flatten()
        result_matrix[i,:] = results_variables_per_factor
    return result_matrix



def __elementTable__(information:np.ndarray, times_symbol:str = "\\cdot", 
                     add_cell_color:bool = True, convergence_color:str = "ForestGreen", not_Convergence_color:str = "BrickRed", cell_alpha:int=5,
                    deactivate_tab_enders:bool=False, number_of_decimals: int = 3):
    """
    # Definition
    This function supposes that somebody give us the results for a factor experiments, and all the methods are flatted in order.
    This is the heart of this script.


    ## Inputs

        ### Required 
        - `information: np.ndarray`         - A row from the matrix results (check the order of the variables)

        ### Optional
        - `times_symbol: string = \cdot`     - Multiplication symbol used over the results in scientific notation
        - `add_cell_color: bool = True`      - Boolean flag to add color to the convergency variable.
        - `convergence_color:string = ForestGreen` - Color for the converged methods
        - `not_convergence_color:string = BrickRed` - Color for the not converged methods
        - `cell_alpha: int = 10`             - Transparency of the convergence cells
        - `deactivate_tab_enders: bool = False` - Delete the TU.TAB_END variable at the end of the tables
        - `number_of_decimals: int = 3` - Number of decimals to be displayed on the Last Gradient norm

    ## Outputs

        - None - The usage of this function is to print the values of the table
   """ 
    
    #String of the row to be created
    s: str = ""

    #Each variable has its own format
    convergence_flags = [information[i] for i in range(0, information.shape[0], 5)]
    iterations = [int(information[i]) for i in range(1, information.shape[0], 5)]
    exe_time = [f"{information[i]: .4f}" for i in range(2, information.shape[0], 5)]
    gradient_norms = [information[i].item() for i in range(3, information.shape[0], 5)]
    divergent_solutions = int(information[-1])

    #The creation of the G-Norm is special, so we must process it
    g_formatted = []
    for g in gradient_norms:

        #If the number is infinite
        if np.isinf(g):
            g_formatted.append("$\infty$")
        
        #If the number is exactly zero
        elif g == 0.0:
           g_formatted.append("0.0") 

        else:
            #If the number it is not a infinite number or zero
            exponent = int(math.floor(math.log10(abs(g))))
            decimal = g / (10 ** exponent)
            if number_of_decimals==4:
                g_formatted.append(f"${decimal: .4f}{times_symbol} 10^{{{int(exponent)}}} $")
            elif number_of_decimals ==5:
                g_formatted.append(f"${decimal: .5f}{times_symbol} 10^{{{int(exponent)}}} $")
            else:
                g_formatted.append(f"${decimal: .3f}{times_symbol} 10^{{{int(exponent)}}} $")
    

    #Creation of the row
    for i in range(0, 3):

        #The convergence is can be partial for the Random - NAMGM (this if handle it)
        if convergence_flags[i] not in [0.0, 1.0]:
            c = f"{convergence_flags[i]: 0.2}" 
            c = f"{{\\scriptsize {c}}}"
            if add_cell_color:
                c+=TU.construct_cell_color(convergence_color, int(convergence_flags[i]*100), not_Convergence_color, cell_alpha)

        #If the method converged completely or not, then
        else: 

            #Symbols of convergency and not convergency
            if bool(convergence_flags[i]):
                c =  f"\\textcolor{{{convergence_color}}}{{{TU.CHECK_SYMBOL}}}"
            else:
                if np.isinf(gradient_norms[i]):
                    c = f"\\textcolor{{{not_Convergence_color}}}{{{TU.CROSS_SYMBOL}}}"
                else:
                    c = TU.CROSS_SYMBOL


            #Adds color to the convergence cell             
            if add_cell_color:
                c+= TU.construct_cell_color(convergence_color, cell_alpha) if bool(convergence_flags[i]) else TU.construct_cell_color(not_Convergence_color, cell_alpha)
        
        #We add the number of times that the method diverged (only in Random-NAMGM) 
        if i==2:       
            c += TU.TAB_LINKER + f"{{\\scriptsize {divergent_solutions}}}"

        #Add the other variables
        s += c + TU.TAB_LINKER + str(iterations[i]) + TU.TAB_LINKER + exe_time[i] + TU.TAB_LINKER + g_formatted[i]
        if i!=2:
            s += TU.TAB_LINKER
        else:
            if not deactivate_tab_enders:
                s += TU.TAB_END
            else:
                s += ""
    return s

def __tableBody__(factors_matrices: list[np.ndarray], times_symbol="\\times", 
                  add_cell_color:bool = True, convergence_color:str = "ForestGreen", not_Convergence_color:str = "BrickRed", cell_alpha:int=10,
                  number_of_decimals:int = 3)->None:
    """ # Definition
    Function to print the results for the robust experiments. The experiments are separated according the factor.
    Each row is created using `elementTable()` function. The original implementation does not include the markers 
    for the configuration in the first 3 rows. Also the headers and bottoms parts were added.
    
    
    ## Inputs:
    
    - factors_matrices: list[np.ndarray] - A list with the results matrices with the format given by the `create_matrix_result`()`. The order
                                            of the matrices must be [SIMPLEST, ORIGINAL, WITH LINE SEARCH] (required)
    - times_symbol:    str: = '\\times'   - Symbol to represent the product in the table
    - add_cell_color:    bool: = True     - Add color to the cell for the convergence variable (apply on all methods and configurations) .
    - convergence_color:str: = 'ForestGreen' - Color of the cell for the converged methods (only visible if add_cell_color is true)
    - not_convergence_color:str: = 'BrickRed' - Color of the cell for the not converged methods (only visible if add_cell_color is true) 
    - cell_alpha:       int: = 10         - Transparency of the convergence cells, the values must be in the interval [0, 100].

    ## Outputs:

    -  None: This functions is just to print the body of the table, and will be invoked to get that part.
    """
    #Assertions to guarantee the correct working of this function
    assert 0<=cell_alpha<=100 and type(cell_alpha)==int, "the alpha of the cell must be an integer and in the interval [0, 100]"
    assert len(factors_matrices)==3, "The list of matrices must have the tree configurations [SIMPLEST, ORIGINAL, WithLS]"
    
    #Row creation (Move first over the f)
    for f in range(factors_matrices[0].shape[0]):
        #print the separation line between factors
        if f==0:
            print("        \\cmidrule(lr){1-15}")
            print(f"        \\multirow{{3}}{{*}}{{$10^{{{f}}}$}}" + TU.TAB_LINKER + __elementTable__(factors_matrices[0][f], times_symbol=times_symbol, deactivate_tab_enders=True, number_of_decimals=number_of_decimals) + TU.TAB_LINKER + "S" +TU.TAB_END)
            print("        " + TU.TAB_LINKER + __elementTable__(factors_matrices[1][f], times_symbol=times_symbol, deactivate_tab_enders=True, number_of_decimals=number_of_decimals) + TU.TAB_LINKER + "O" +TU.TAB_END)
            print(        "" + TU.TAB_LINKER + __elementTable__(factors_matrices[2][f], times_symbol=times_symbol, deactivate_tab_enders=True, number_of_decimals=number_of_decimals) + TU.TAB_LINKER + "W"+ TU.TAB_END)
            continue
        elif f==1:
           print("        \\cmidrule(lr){1-15}")            
        else: 
            print("        \\cmidrule(lr){1-14}")
        
        #We print the body of the table.
        print(f"        \\multirow{{3}}{{*}}{{$10^{{{f}}}$}}" + TU.TAB_LINKER + __elementTable__(factors_matrices[0][f], times_symbol=times_symbol, number_of_decimals=number_of_decimals))
        print("        " + TU.TAB_LINKER + __elementTable__(factors_matrices[1][f], times_symbol=times_symbol, number_of_decimals=number_of_decimals))
        print("        " + TU.TAB_LINKER + __elementTable__(factors_matrices[2][f], times_symbol=times_symbol, number_of_decimals=number_of_decimals))
    return None 


def __headerTable__(function_name:str, scalebox_value: float = 0.65)->None:
    
    assert 0.0 <= scalebox_value <= 1, "The scalebox_value value must be on the interval [0, 1]"

    header_standard_A:str = """
\\begin{table}[H]
    \\centering
    \\scalebox{0.65}{
    \\begin{tabular}{|c|c|ccc|c|ccc|c|c|ccc|c|}
        \\toprule"""
    
    header_file_name = f"""        \\multicolumn{{15}}{{|c|}}{{\\cellcolor{{PerlWhite}}\\textbf{{Problem Name: {function_name}}}}}"""+TU.TAB_END

    header_standard_B:str = """        \\cmidrule(lr){1-15}
        \\multirow{2}{*}{$\\bs{F}$} & \\multicolumn{4}{|c}{NAMGM-AMG}  & \\multicolumn{4}{|c}{NAMGM-\\textit{Queue}} & \\multicolumn{5}{|c|}{NAMGM-\\textit{Random}}& \\multirow{2}{*}{Config}\\\\
        \\cmidrule(lr){2-5} \cmidrule(lr){6-9}\cmidrule(lr){10-14} 
        & C & Iters & Time (s) & Last Norm & C & Iters & Time (s) & Last Norm & C & NC & Iters & Time (s) & Last Norm & \\\\"""

    print(header_standard_A)
    print(header_file_name)
    print(header_standard_B)
    return None

def __bottomTable__(caption:str = ""):
    lastPart = """        \\cmidrule(lr){1-14}
    \end{tabular}}"""
    captionPart = f"""
    \caption{{{caption}}}
\end{{table}}"""
    print(lastPart+captionPart)


def generateTable(function_name:str, factors_matrices: list[np.ndarray],scalebox_value: float = 0.65, times_symbol="\\times", 
                  add_cell_color:bool = True, convergence_color:str = "ForestGreen", not_Convergence_color:str = "BrickRed", cell_alpha:int=10, 
                  number_of_decimals:int = 3)->None:

    __headerTable__(function_name, scalebox_value)
    __tableBody__(factors_matrices, times_symbol, add_cell_color, convergence_color, not_Convergence_color, cell_alpha, number_of_decimals)
    __bottomTable__()

    return None

def main()->None:

    #Get the values from the parser
    args = parser.parse_args() 

    #Directory of the results original
    directory_simplest = args.simplest + "*.csv"
    directory_original = args.original + "*.csv"
    directory_addedLS = args.withLs + "*.csv"
    v = bool(args.verbose)
    decimals = 3
    verbose = 2 if v else 0

    #Read all the results of the dataframes
    dataframes_orig, names_frames, problem_name = get_ListOfDF(directory_original, verbose)
    dataframes_LS, names_frames = get_ListOfDF(directory_addedLS, verbose)[0:-1]
    dataframes_SP, names_frames = get_ListOfDF(directory_simplest, verbose)[0:-1]

    #Results in the row format
    matrix_simplest = create_matrix_results(dataframes_SP)  
    matrix_original = create_matrix_results(dataframes_orig) 
    matrix_addedLS = create_matrix_results(dataframes_LS) 

    #List of matrix results
    list_matrices = [matrix_simplest, matrix_original, matrix_addedLS]
    generateTable(problem_name, list_matrices, 0.58, "\\cdot", number_of_decimals=decimals)

    return

if __name__ =="__main__":
    main()