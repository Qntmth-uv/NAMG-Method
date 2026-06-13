import table_utils as TU
import numpy as np
import pandas as pd
import argparse
import glob 
from pathlib import Path

parser = argparse.ArgumentParser(description="Paths where the results are saved")

parser.add_argument("-s", "--simplest", help="Path to simplest directory")
parser.add_argument("-o", "--original", help="Path to original directory")
parser.add_argument("-w", "--withLS", help="Path to addedLS directory")

def get_ListOfDF(directory:str, verbose:int = 0):
    # Read all the results tables in the directory
    csv_files = glob.glob(directory)

    # Sort the tables by name
    csv_files = sorted(csv_files)

    # Read all the csv files from the directory, and save we save them in a list
    dataframes = [pd.read_csv(f) for f in csv_files]
    names_frames = [Path(csv).stem.split('_')[0] for csv in csv_files]

    #We print the name of the read tables if it's required
    if verbose in [-1, 0]: print(names_frames)
    if verbose in [0,1]: print("Read path: ", directory)

    # We print the tables read
    if verbose == 0:
        for i in range(0, len(csv_files)):
            print(f"|{i}| The table has been read it from the directory: {csv_files[i]}")
        print("-"*80)
    if verbose in [-1,0,1]:
        print(f"{len(dataframes)} tables has been read.")

    return dataframes, names_frames


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def construct_matrixResults(list_dataframes: list[pd.DataFrame])->np.ndarray:
    """
    ## Usage
    Function to get the values from the dataframes. This function create a Numpy
    array of 3 dimensions, the measured variable (first dimension), the number of 
    problem, and the method. 

    ### Input. 
        - `list_dataframes`: list[pd.DataFrame] - 'Descriptive name'

    ### Output.
        - `result_tensor`: np.ndarray - Tensor containing all the results.

    The order of the variables is the same (check remarks for more information),
    
    ### Remarks.
    The measured variables were (in the given order in the csv files):

        1 - Iterations. The number of iterations that the algorithm took
        2 - Execution time. Time that the algorithm took.
        3 - LastNorm. The gradient norm from the last element in the sequence.
        4 - Iterations per second - Iterations/Execution_time
        5 - Convergence. If the algorithm converged in the given iterations and minimal 
        acceptable gradient.

    The problem solved are described in the ```problem_dimension``` variable. And the order
    of the problems are

        0 - AMG. NAMGM/Accelerated minimal gradient set of vectors
        1 - Grads. NAMGM/Gradient set of vectors
        2 - Random. NAMGM/Random set of vectors
        3 - Newton. ¨¨
        4 - BFGS. ¨¨
        5 - GDLS. Gradient descent with Line Search (this if fix for all problems)          

    ### Example usage.
    To observe the results of number of iterations use `result_tensor[0, :, :]`. 
    Then the row dimension are the problems and the columns are the different methods.
    """
    # We have 5 methods and 5 variables, then in total we have N \times 5 elements in the dataframe
    result_tensor = np.zeros((5, len(list_dataframes), 6)) #(Measured Variables, Problems, Methods)

    #Obtain the information of each method (by rows)
    for i in range(0, len(list_dataframes)):
        convergence = list_dataframes[i]["Archived Convergence"].to_numpy()
        iterations = list_dataframes[i]["iterations"].to_numpy()
        grads = list_dataframes[i]["Last Gradient"].to_numpy()
        exc = list_dataframes[i]["Execution time"].to_numpy()
        ittPSec = list_dataframes[i]["Iterations per Second"].to_numpy()

        #Setting the information in the result matrix
        result_tensor[:, i, :] = [iterations, exc, grads, ittPSec, convergence]

    return result_tensor

def change_orderOfMethods(result_tensor:np.ndarray, new_order:list[int] = [0, 1, 2, 3, 4, 5]):
    """
    # Usage
    Function to modify the order of the presented method in the last dimension.
    The original order of the methods is.
    
    - 0 - NAMGM/Using the AMG set of vectors
    - 1 - NAMGM/Using the Queue set of vectors
    - 2 - NAMGM/Using the Random set of vectors
    - 3 - Modified Newton's method
    - 4 - BFGS method
    - 5 - Gradient Descent with Line Search
    
    ## Input.
        - `result_tensor`: np.ndarray - 'Descriptive name'
        - `new_order`: list[int] - New proposed order

    ## Output.
        - `result_tensor`: np.ndarray - Same tensor results but a new method other    
        
    ## Remark. 
    The use of line search (LS) does not modify the order, this is the given order 
    for all the different kind of experiments with or without the use of LS.
    """

    assert len(new_order) == result_tensor.shape[2], "The given reorder does not match with the dimension of the methods"
    reOrdered_matrix = result_tensor[:, :, new_order]
    return reOrdered_matrix

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Function to print the results depending on the value on the column
def format_column(value:any, variable_int: int, use_symbols:bool, use_colors:bool, time_in_scientific:bool = True,
                  final_element:bool = False, is_local_best:bool = False, is_global_best:bool = False, it_converged:bool = True)->str:
    """
    # Usage


    ## Inputs:

        ### Required:
            - value(any): Variable value (integer or float)
            - variable_int(int): ID of the variable.
            - use_symbols(bool): Use the symbols defined in the variables `CHECK_SYMBOL`and `CROSS_SYMBOL`. 
            - use_colors(bool): Add colors to the convergence symbols (the colors are fix, for convergence is used 'ForestGreen' and 'BrickRed' for the other).

        ### Optional:
            - final_element(bool) = False: Boolean variable to indicate if the value is at the end of the constructed table.
            - is_global_best(bool) = False: Boolean variable to indicate if the value is the best on all configurations.
            - it_converged(bool) = True: Boolean variable to indicate if the method converged.
    
    ## Outputs:

        - s(str): String variable constructed according to the corresponding variable.
        
    
    ## Remarks.
    The number of variable changes the format of the value element. The order of the methods is the following:

    0. Number of iterations
    1. Time execution
    2. Last gradient norm
    3. Iterations per second
    x. Convergence.

    Where X is any other integer different of 0,1,2,3. Also we dont implement one function for the different use
    of bolding, due that it is not general enough.
    """ 

    #String where save the element of the table
    s: str = ""   

    # If the method diverged, then we add a special flag to indicate it. And returned immediately.
    if np.isinf(value): 
        s += TU.TEXT_C("BrickRed", TU.CROSS_SYMBOL)
        s = TU.construct_cell_color("BrickRed", 10) + s if use_colors else s

    #0. Number of iterations variable.
    elif variable_int == 0:
        
        #In Random-NAMGM the number of iterations is a mean. Therefore, can be a real value.
        if value.is_integer():
            value = int(value)
            s += f"{value: d}"

        #If it is not an integer, then we use the format INTEGER.XXX. Where X is a digit.
        else:
            s += f"{value: .2f}"

        #If it is a local best the and global, we use the special 
        if is_local_best and is_global_best:
            s = TU.add_bold_to_element(s, False, is_best_value=True, it_converged=it_converged)

        #If it is a local best, then we use normal bold.
        elif is_local_best:
            s = TU.add_bold_to_element(s, False, is_best_value=False, it_converged=it_converged)

        else: pass

        
    #1,2. Time execution and Last gradient norm variables
    elif variable_int in [1,2]: 
        #If it is a local best the and global, we use the special 
        if is_local_best and is_global_best:
            s += TU.scientific_notation_converter(value, remove_math_mode=True)
            if it_converged:
                s = TU.add_bold_to_element(s, True, is_best_value=True, add_math_mode=True, it_converged=it_converged)
            else:
                s = f"${s}$"                        
        #If it is a local best, then we use normal bold.
        elif is_local_best:
            s += TU.scientific_notation_converter(value, remove_math_mode=True)
            if it_converged:
                s = TU.add_bold_to_element(s, True, is_best_value=False, add_math_mode=True)
            else:
                s = f"${s}$"
        else: 
            s +=TU.scientific_notation_converter(value, remove_math_mode=False)

    #3. Number of iterations per second variable
    elif variable_int == 3: 

        #Format: Integer.XX, where XX are digits
        s += f"{value: .2f}"

        #If it is a local best the and global, we use the special 
        if is_local_best and is_global_best:
            s = TU.add_bold_to_element(s, False, is_best_value=True, it_converged=it_converged)

        #If it is a local best, then we use normal bold.
        elif is_local_best:
            s = TU.add_bold_to_element(s, False, is_best_value=False, it_converged=it_converged)
        else: pass

    #4. Convergence flag 
    else:
        #Boolean variable to indicate if the method converged or not
        convergence_flag:bool = bool(value)

        #If it is the Random method were the convergence is a probability, then we use a special format.
        if value != 0.0 and value != 1.0:
            s += "{\\scriptsize "+TU.TEXT_C(f"ForestGreen!{int(value*100)}!gray", f"{value:1.2f}")+"}"

        #If it's a boolean variable, then we use the boolean variables.
        else:

            #It converged
            if convergence_flag:
                if use_colors:s += TU.construct_cell_color("ForestGreen", 10)
                if use_symbols: s+= TU.TEXT_C('ForestGreen', TU.CHECK_SYMBOL)

            #Do not converged
            else:
                if use_colors: s += TU.construct_cell_color("gray", 10) #If we add colors
                if use_symbols: s += TU.TEXT_C('gray', TU.CROSS_SYMBOL) #If we use symbols.
    

    #Add the respective linker
    s += TU.TAB_LINKER if not final_element else TU.TAB_END
    return s

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Convergence table style.
class experiment1_table(TU.LaTeXTable):
    def __init__(self, scalebox = 1, colors = True, extra_indentation = 0, columns_format:str = ""):
        super().__init__(scalebox, colors, extra_indentation)

        #Tabular objects
        self.columns_format = columns_format
        self.column_names = ["GDLS", "AMG", "Queue", "Random", "BFGS", "Newton"]
        self.configurations_names = ["Simplest", "Original", "With LS", "Standard"]
        self.variables_order = {0: "Iterations", 1: "Time", 2:"Last Grad", 3: "It/sec", 4: "Convergence"}
        self.original_order = self.variables_order.keys()
        self.used_order = [4, 0, 1, 2, 3] #First the convergence

        #Convergence parameters
        self.add_colors:bool = colors
        self.convergence_color: str = "ForestGreen"
        self.not_convergence_color:str = "BrickRed"
        self.convergence_alpha:int = 5


    def __headerTabular__(self, function_name:str = "", dim_problem:int = 2)->None:

        #Header Objects
        self.function_name = function_name
        self.dim_problem = dim_problem

        #Definition of the headers
        main_header = self.function_name + f" (Dim: {self.dim_problem})"
        main_header = main_header + TU.construct_cell_color("orange", 20) if self.dim_problem > 1000 else main_header
        header_1 = [self.init_tabular("ccccccc"), TU.T_RULE, TU.MULTI_COL(7, "c", main_header)+TU.TAB_END]
        header_2 = [TU.M_RULE, "Config & Algorithm & C &Iters & Time (s) & Last Norm & It/sec\\\\", TU.M_RULE]
        header = header_1 + header_2

        #Definition of the indentation levels according to the number of elements in the header. 
        idnttn_levels = [self.last_indentation_level]+([self.last_indentation_level+1]*5)
        
        #Print the corresponding line
        for (e, i) in zip(header, idnttn_levels): print(self.einTab*i + e) 
        self.last_indentation_level +=1


    def __bodyTabular__(self, problem_data:np.ndarray)->None:
        """
        ### Remark.
        We suppose that `problem_data` contains the information inf the following order:
        `[simplest, original, addedLS]` of the current problem.
        """

        #Current indentation 
        indentation_level = self.einTab * self.last_indentation_level

        #Get the index of the best local and global values
        best_values_packages = TU.values_and_index_best(problem_data, self.used_order)
        index_local, index_global = best_values_packages[1], best_values_packages[-1]
        #print("------->", index_global, ">>>", index_global)

        #Creation of the row of the table
        #Move over the configurations ()
        for k in range(len(problem_data)):

            #Movement over the methods
            for j in range(problem_data[k].shape[1]):

                #If the method is [GDLS, BFGS, then do not print]
                if j in [0,4]: pass

                #Otherwise
                else:
                    #The method converged? (flag used to avoid HL no convergent values)
                    convergence_flag_method = bool(problem_data[k, -1, j])
                    if j==1:
                        extra_element = TU.MULTI_ROW(4, self.configurations_names[k])
                    else:
                        extra_element = " "                 

                    #First element in the tab                    
                    print(indentation_level + extra_element + TU.TAB_LINKER + self.column_names[j]+TU.TAB_LINKER, end="")

                    #Movement over the variables
                    for i in self.used_order:
                        
                        #IDENTIFIER INSIDE THE TABLE
                        local_id = [k, j, i]
                        best_local_flag = local_id in index_local
                        best_global_flag = local_id in index_global

                        #Get the value in the adequate format
                        if i == self.used_order[-1]:
                            print(format_column(problem_data[k, i, j], i, use_symbols=True, use_colors=self.add_colors, final_element=True,
                                                is_local_best=best_local_flag, is_global_best=best_global_flag, 
                                                it_converged=convergence_flag_method))
                        else:
                            print(format_column(problem_data[k, i, j], i, use_symbols=True, 
                                                use_colors=self.add_colors, is_local_best=best_local_flag, is_global_best=best_global_flag,
                                                it_converged=convergence_flag_method), end="")
            print(indentation_level+TU.C_RULE(2,7))
            
        #Print methods that are equal in several executions (SGLS, BFGS)
        for j in range(problem_data[0].shape[1]):
            if j not in [0,4]: pass
            else:
                #Constructed string
                if j==0:
                    extra_element = TU.MULTI_ROW(2, self.configurations_names[-1])
                else:
                    extra_element = ""                 
                print(indentation_level + extra_element + TU.TAB_LINKER + self.column_names[j], end = TU.TAB_LINKER)
                
                #Movement over the variables
                for i in self.used_order:

                    #ID INSIDE THE TABLE
                    local_id = [k, j, i]
                    best_local_flag = local_id in index_local
                    best_global_flag = local_id in index_global

                    #Format of the element in the table
                    if i == self.used_order[-1]:
                        print(indentation_level + format_column(problem_data[k, i, j], i, use_symbols=True, use_colors=self.add_colors, final_element=True,
                                    is_local_best=best_local_flag, is_global_best=best_global_flag, it_converged=convergence_flag_method))
                    else:
                        print(indentation_level + format_column(problem_data[k, i, j], i, use_symbols=True, 
                                    use_colors=self.add_colors, is_local_best=best_local_flag, is_global_best=best_global_flag,
                                    it_converged=convergence_flag_method), end="")
        print(indentation_level + TU.B_RULE)

    def __bottomTabular__(self,)->None:
        bottom = [self.end_tabular]
        idbttn_levels = [self.last_indentation_level]
        self.last_indentation_level -= 1
        for (e, i) in zip(bottom, idbttn_levels):print(self.einTab * i + e)
        pass

    def generateTabular(self, function_name:str, dimension: int, problem_data:np.ndarray)->None:
        self.__headerTabular__(function_name, dimension)
        self.__bodyTabular__(problem_data)
        self.__bottomTabular__()

# Dictionary of the dimensions of the testet problems
problem_dimensions = {
    "ARGLINA": 200,
    "BARD": 3,
    "BEALE" : 2,
    "BRKMCC": 2,
    "BROWNAL": 200,
    "BROWNBS": 2,
    "BROWNDEN": 4,
    "CHNROSNB": 50,
    "CLIFF": 2,
    "CUBE": 2,
    "DECONVU": 61,
    "DENSCHNA": 2,
    "DENSCHNB": 2,
    "DENSCHNC": 2,
    "DENSCHND": 2,
    "DENSCHNF": 2,
    "DIXON3DQ": 10000,
    "EIGENALS": 2550,
    "EIGENBLS": 2550,
    "ENGVAL2": 3,
    "EXTROSNB": 100,
    "FLETCBV2": 5000,
    "FLETCHCR": 1000,
    "GENHUMPS": 5000,
    "HAIRY": 2,
    "HEART6LS": 6,
    "HELIX": 3,
    "HILBERTA": 2,
    "HILBERTB": 10,
    "HIMMELBB": 2,
    "HIMMELBH": 2,
    "HUMPS": 2,
    "JENSMP": 2,
    "KOWOSB": 4,
    "LOGHAIRY": 2,
    "MANCINO": 100,
    "MARATOSB": 2,
    "MEXHAT": 2,
    "PALMER1C": 8,
    "PALMER2C": 8,
    "PALMER3C": 8,
    "PALMER4C": 8,
    "PALMER5C": 6, 
    "PALMER6C": 8,
    "PALMER7C": 8,
    "PALMER8C": 8,
    "ROSENBR": 2,
    "SINEVAL": 2,
    "SISSER": 2,
    "TOINTQOR": 50,
    "VARDIM": 200,
    "WATSON": 31,
    "YFITU": 3,
    "BIGGS6": 6,
    "BOX3": 3,
    "BROWNBS": 2,
    "BROWNDEN": 4,
    "GAUSSIAN": 3,
    "GULF": 3,
    "PENALTY1": 1000,
    "PENALTY2": 1000,
    "TRIGON1": 10_000,
    "WOODS": 1000,
}

def main()->int:

    #Get the values from the parser
    args = parser.parse_args() 

    #Directory of the results original
    directory_simplest = args.simplest + "*.csv"
    directory_original = args.original + "*.csv"
    directory_addedLS = args.withLS + "*.csv"

    #Read all the results of the dataframes
    dataframes_SP, names_frames = get_ListOfDF(directory_simplest, -1)
    dataframes_orig, names_frames = get_ListOfDF(directory_original, -1)
    dataframes_LS, names_frames = get_ListOfDF(directory_addedLS, -1)

    #Get the matrix of results
    result_tensor_simplest = construct_matrixResults(dataframes_SP)
    result_tensor_original = construct_matrixResults(dataframes_orig)
    result_tensor_addedLS = construct_matrixResults(dataframes_LS)

    #Reorder the matrix results
    result_tensor_simplest = change_orderOfMethods(result_tensor_simplest, [5, 0, 1, 2, 4, 3])
    result_tensor_original = change_orderOfMethods(result_tensor_original, [5, 0, 1, 2, 4, 3])
    result_tensor_addedLS = change_orderOfMethods(result_tensor_addedLS, [5, 0, 1, 2, 4, 3])

    #Assertion that all results matrices has the same number of problems
    assert result_tensor_addedLS.shape[1] == result_tensor_simplest.shape[1] and result_tensor_simplest.shape[1] == result_tensor_original.shape[1], "The results matrices does not have the same dimension"
    nProblems:int = result_tensor_simplest.shape[1]
    
    #Get the result table for the current problem
    
    for i in range(nProblems):
        last_index = 0
        #List of matrices result1s
        info = np.array([result_tensor_simplest[:,i,:], 
                result_tensor_original[:,i,:],
                result_tensor_addedLS[:,i,:]])    

        #Configuration of the table.
        exp1 = experiment1_table(extra_indentation=2)
        name = names_frames[i]
        dim = problem_dimensions[name]
        exp1.generateTabular(name, dim, info)

        print("---"*20, end="\n\n")


if __name__ =="__main__":
    main()