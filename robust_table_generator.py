import numpy as np
import pandas as pd
import glob 
import math
from pathlib import Path

TAB_LINKER = " & "
TAB_END = "\\\\"
CHECK_SYMBOL = "\\checkmark"
CROSS_SYMBOL = "\\texttimes"

def get_ListOfDF(directory:str, verbose:int = 0):
    # Read all the results tables in the directory
    csv_files = glob.glob(directory)

    # Sort the tables by name
    csv_files = sorted(csv_files)

    # Read all the csv files from the directory, and save we save them in a list
    dataframes = [pd.read_csv(f) for f in csv_files]
    names_frames_complete = [Path(csv).stem.split('.csv')[0] for csv in csv_files] 

    #We print the name of the read tables if it's required
    if verbose in [-1, 0]: 
        print(names_frames_complete)
        print(names_frames_complete)
    if verbose in [0,1]: print("Read path: ", directory)

    # We print the tables read
    if verbose == 0:
        for i in range(0, len(csv_files)):
            print(f"|{i}| The table has been read it from the directory: {csv_files[i]}")
        print("-"*80)
    if verbose in [-1,0,1]:
        print(f"{len(dataframes)} tables has been read.")

    return dataframes, names_frames_complete 

def create_matrix_results(dataframes_list: list[pd.DataFrame])->np.ndarray:
    """
    # Definition
    This function create an np.ndarray element class. Each element in the dataframe
    has the results from the the three methods, whose have the following order:

    1. NAMGM/AMG
    2. NAMGM/Queue
    3. NAMGM/Random

    And the order of the columns is 

    1. Convergence 
    2. Number of iterations taken
    3. Execution time
    4. Last gradient norm of the sequence

    Observe that we are not longer measuring the 'iterations per second'. 
    
    # Inputs
    The big difference is that the creation of each matrix is based in one problem and we not have
    to modify the order of the variables (this is the big difference btw `robust_tables` and `creation_tables`).
    
    - dataframes_list: list[pd.dataframes] - The list of dataframes of the different factors for the problem


    # Output

    - matrix_results: np.ndarray - Numpy array with the results for the Configuration/Problem

    """
    # We have 5 methods and 5 variables, then in total we have N \times 5 elements in the dataframe
    result_matrix = np.zeros((len(dataframes_list), 12)) #(Number of factors, Variables (4 variables, but 3 methods))

    #Picking of the information of each method (picking in rows)
    for i in range(0, len(dataframes_list)):
        results_variables_per_factor: np.ndarray = dataframes_list[i].to_numpy().flatten()
        result_matrix[i,:] = results_variables_per_factor
    return result_matrix

def construct_cell_color(color1:str, alpha1:int, color2:str =None, alpha2:int=None)->str:
    """# Usage
    
    Function to construct the color cell on a LaTeX table. If color2 and alpha 2 are
    not provided, it still works.
    
    ## Input:
        - ``color1``: string - First (or main) color of the cell
        - ``alpha1``:   int  - Transparency (or percentage of the first color) 
        - ``color1``(optional): string - Second color of the cell
        - ``alpha2``(optional):   int  - Transparency of the combined color 

    ## Output:
        - ``s``: string - LaTeX command to color the cell
    """
    final_color:str = f"{color1}!{alpha1}"
    if color2:
        final_color += f"!{color2}"
        if alpha2:
            final_color += f"!{alpha2}"
    return f"\\cellcolor{{{final_color}}}"

def __elementTable__(information:np.ndarray, times_symbol:str = "\\times", 
                     add_cell_color:bool = True, convergence_color:str = "ForestGreen", not_Convergence_color:str = "BrickRed", cell_alpha:int=10,
                     deactivate_tab_enders:bool=False):
    """Suppose that some one give us the results for a factor, and all the methods are flatted in order."""
    s = ""
    #Each variable has its own format
    convergence_flags = [bool(information[i]) for i in range(0, information.shape[0], 4)]
    iterations = [int(information[i]) for i in range(1, information.shape[0], 4)]
    exe_time = [f"{information[i]: .4f}" for i in range(2, information.shape[0], 4)]

    gradient_norms = [information[i].item() for i in range(3, information.shape[0], 4)]
    g_formatted = []
    for g in gradient_norms:
        exponent = int(math.floor(math.log10(abs(g))))
        decimal = g / (10 ** exponent)
        g_formatted.append(f"${decimal: .3f}{times_symbol} 10^{{{int(exponent)}}} $")
    
    for i in range(0, 3):
        c = CHECK_SYMBOL if convergence_flags[i] else CROSS_SYMBOL            
        if add_cell_color:
            c+= construct_cell_color(convergence_color, cell_alpha) if convergence_flags[i] else construct_cell_color(not_Convergence_color, cell_alpha)
        s += c + TAB_LINKER + str(iterations[i]) + TAB_LINKER + exe_time[i] + TAB_LINKER + g_formatted[i]
        if i!=2:
            s += TAB_LINKER
        else:
            if not deactivate_tab_enders:
                s += TAB_END
            else:
                s += ""
    return s

def __tableBody__(factors_matrices: list[np.ndarray], times_symbol="\\times", 
                  add_cell_color:bool = True, convergence_color:str = "ForestGreen", not_Convergence_color:str = "BrickRed", cell_alpha:int=10)->None:
    """ # Definition
    Function to print the results for the robust experiments. The experiments are separated according the factor.
    Each row is created using `elementTable()` function. The original implementation does not include the markers 
    for the configuration in the first 3 rows. Also the headers and bottoms parts were added.
    
    
    ## Inputs:
    
    - factors_matrices: list[np.ndarray] - A list with the results matrices with the format given by the `create_matrix_result`()`. The order
                                            of the matrices must be [SIMPLEST, ORIGINAL, WITH LINE SEARCH] (requiered)
    - times_symbol:    str: = '\\times'   - Symbol to represent the product in the table
    - add_cell_color:    bool: = True     - Add color to the cell for the convergence variable (apply on all methods and configurations) .
    - convergence_color:str: = 'ForestGreen' - Color of the cell for the converged methods (only visible if add_cell_color is true)
    - not_convergence_color:str: = 'BrickRed' - Color of the cell for the not converged methods (only visible if add_cell_color is true) 
    - cell_alpha:       int: = 10         - Transparency of the convergence cells, the values must be in the interval [0, 100].

        ## Outputs:

    -  None: This functions is just to print the body of the table, and will be invoked to get that part.
    


    """
    #Assertions to garantie the correct working of this function
    assert 0<=cell_alpha<=100 and type(cell_alpha)==int, "the alpha of the cell must be an integer and in the interval [0, 100]"
    assert len(factors_matrices)==3, "The list of matrices must have the tree configurations [SIMPLEST, ORIGINAL, WithLS]"
    
    #Row creation (Move first over the f)
    for f in range(factors_matrices[0].shape[0]):

        #print the separation line between factors
        if f==0:
            print("\\cmidrule(lr){1-14}")
            print(f"\\multirow{{3}}{{*}}{{$10^{{{f}}}$}}" + TAB_LINKER + __elementTable__(factors_matrices[0][f], times_symbol=times_symbol, deactivate_tab_enders=True) + TAB_LINKER + "S" +TAB_END)
            print("" + TAB_LINKER + __elementTable__(factors_matrices[1][f], times_symbol=times_symbol,deactivate_tab_enders=True) + TAB_LINKER + "O" +TAB_END)
            print("" + TAB_LINKER + __elementTable__(factors_matrices[2][f], times_symbol=times_symbol, deactivate_tab_enders=True) + TAB_LINKER + "W"+ TAB_END)
            continue
        elif f==1:
           print("\\cmidrule(lr){1-14}")            
        else: 
            print("\\cmidrule(lr){1-13}")
        #We print the body of the table.
        print(f"\\multirow{{3}}{{*}}{{$10^{{{f}}}$}}" + TAB_LINKER + __elementTable__(factors_matrices[0][f], times_symbol=times_symbol))
        print("" + TAB_LINKER + __elementTable__(factors_matrices[1][f], times_symbol=times_symbol))
        print("" + TAB_LINKER + __elementTable__(factors_matrices[2][f], times_symbol=times_symbol))
    return None 


def __headerTable__(function_name:str, scalebox_value: float = 0.65)->None:
    assert 0.0 <= scalebox_value <= 1, "The scalebox_value value must be on the interval [0, 1]"

    header_standard_A:str = """
\\begin{table}[H]
    \\centering
    \\scalebox{0.65}{
    \\begin{tabular}{|c|c|ccc|c|ccc|c|ccc|c|}
        \\toprule"""
    
    header_file_name = f"""        \\multicolumn{{14}}{{|c|}}{{\\cellcolor{{PerlWhite}}\\textbf{{Problem Name: {function_name}}}}}"""+TAB_END

    header_standard_B:str = """        \\cmidrule(lr){1-14}
        \\multirow{2}{*}{$\\bs{F}$} & \\multicolumn{4}{|c}{NAMGM-AMG}  & \\multicolumn{4}{|c}{NAMGM-\\textit{Queue}} & \\multicolumn{4}{|c|}{NAMGM-\\textit{Random}}& \\multirow{2}{*}{Config}\\\\
        \\cmidrule(lr){2-5} \cmidrule(lr){6-9}\cmidrule(lr){10-13} 
        & C & Iters & Time (s) & Last Norm & C & Iters & Time (s) & Last Norm & C & Iters & Time (s) & Last Norm & \\\\"""

    print(header_standard_A)
    print(header_file_name)
    print(header_standard_B)
    return None

def __bottomTable__():
    lastPart = """        \\cmidrule(lr){1-13}
    \end{tabular}}
    \caption{}
\end{table}"""
    print(lastPart)


def generateTable(function_name:str, factors_matrices: list[np.ndarray],scalebox_value: float = 0.65, times_symbol="\\times", 
                  add_cell_color:bool = True, convergence_color:str = "ForestGreen", not_Convergence_color:str = "BrickRed", cell_alpha:int=10)->None:

    __headerTable__(function_name, scalebox_value)
    __tableBody__(factors_matrices, times_symbol, add_cell_color, convergence_color, not_Convergence_color, cell_alpha)
    __bottomTable__()

    return None

def main()->None:
    

    #Directory of the results original
    directory_original = "csvs/results/robust/original/*.csv"
    directory_simplest = "csvs/results/robust/simplest/*.csv"
    directory_addedLS = "csvs/results/robust/addedLS/*.csv"


    #Read all the results of the dataframes
    dataframes_orig, names_frames = get_ListOfDF(directory_original, 0)
    dataframes_LS, names_frames = get_ListOfDF(directory_addedLS, 0)
    dataframes_SP, names_frames = get_ListOfDF(directory_simplest, 0)

    matrix_simplest = create_matrix_results(dataframes_SP)  
    matrix_original = create_matrix_results(dataframes_orig) 
    matrix_addedLS = create_matrix_results(dataframes_LS) 

    #List of matrix results
    list_matrices = [matrix_simplest, matrix_original, matrix_simplest]


    generateTable("WATSON", list_matrices, 0.65, "\\cdot")
    #__tableBody__(list_matrices, "\\cdot")

    return



if __name__ =="__main__":
    main()