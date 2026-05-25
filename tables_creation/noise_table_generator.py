import numpy as np
import pandas as pd
import glob 
import math
import argparse
from pathlib import Path
import os

#Own Libraries
import table_utils as TU
from mixing_information import read_csvs, existing_folders


"""
# Script intention

This script is intended create all the tables needed to present the results of the noise experiments. It uses as basis the script
'robust_table_generator.py' with an small modification in the convergence flag. Now each method contains a new column to indicate the
number of times that the method diverged.

The big difference with the 'robust_table_generator' is that now, instead of factors, the rows are problems. 

### Usage commands 
python3 robust_table_generator.py -o ../csvs/results/robust/original/BOX3/ -w ../csvs/results/robust/addedLS/BOX3/ -s ../csvs/results/robust/simplest/BOX3/ > tab.tex
python3 robust_table_generator.py -o ../csvs/results/robust/original/WOODS/ -w ../csvs/results/robust/addedLS/WOODS/ -s ../csvs/results/robust/simplest/WOODS/ > tab.tex

# Modification to the SGLS method.
Given that the SGLS method is executed in each configuration, we have 3 repeted values of the SGLS method. We can compute again 
the main of those results and present such result
"""

#Parser variables
parser = argparse.ArgumentParser(description="Path to the directories were to search")

#Variables (paths of the factors results) 
parser.add_argument("-p", "--problems_path", help="Path were all the results of the noise/area robustness were saved.")
parser.add_argument("-sgls","--sgls", action="store_true", help="Create the body of the elements for the SGLS table.")

def create_matrix_results(dataframes_list: list[pd.DataFrame])->np.ndarray:
    """
    ## Definition
    This function create an `np.ndarray` element class. Each element in the dataframe
    has the results from the the fourth methods, whose have the following order:
    
    0. NAMGM/AMG
    1. NAMGM/Queue
    2. NAMGM/Random
    3. SGLS

    And the order of the columns is 

    1. Convergence 
    2. Number of iterations taken
    3. Execution time
    4. Last gradient norm of the sequence
    5. Number of times that the method not diverged
    
    ### Inputs
    The big difference is that the creation of each matrix is based in one problem and we not have
    to modify the order of the variables (this is the big difference btw `robust_tables` and `creation_tables`).
    
        - dataframes_list: list[pd.dataframes] - The list of dataframes of the different factors for the problem

    ### Output

    - matrix_results: np.ndarray - Numpy array with the results for the Configuration/Problem
    - array_results: np.ndarray - A Numpy array with the results of the SGLS/SG
    """
    # We have 4 methods (N) and 5 variables, then in total we have N \times 5 elements in the dataframe
    results_sgls = np.zeros(5, dtype=float) #Special variable to the results of the SGLS
    result_matrix = np.zeros((len(dataframes_list), 5*3)) #(Number of factors, Variables (5 variables * 3 methods))

    #Picking of the information of each method (picking in rows)
    for i in range(0, len(dataframes_list)):
        results_variables_per_factor = dataframes_list[i].to_numpy()        
        result_matrix[i,:] = results_variables_per_factor[0:-1, :].flatten()
        results_sgls += results_variables_per_factor[-1,:]

    #Post process
    results_sgls /= len(dataframes_list)
    return result_matrix, results_sgls

class latexTable:

    def __init__(self, name:str, scalebox:float = 0.58, colors:bool = True, extra_indentation:int = 0):

        #Assertion of values
        assert 0.0 <= scalebox <= 1, "The scalebox_value value must be on the interval [0, 1]"
        #assert 0<=cell_alpha<=100 and type(cell_alpha)==int, "the alpha of the cell must be an integer and in the interval [0, 100]"

        #Header properties
        self.function_name:str = name

        #Scientific notation parameters
        self.scalebox_value = scalebox
        self.decimals: int = 3
        self.times_symbol:str = "\\cdot"
        self.time_in_scientific: bool = False

        #Convergence parameters
        self.add_colors:bool = colors
        self.convergence_color: str = "ForestGreen"
        self.not_convergence_color:str = "BrickRed"
        self.convergence_alpha:int = 5

        #Constants
        self.einTab = "    "
        self.init_tabular = lambda s: "\\begin{tabular}"+f"{{{s}}}" 
        self.end_tabular = "\\end{tabular}"
        self.original_indentation = extra_indentation
        self.last_indentation_level:int = extra_indentation
        self.idnttn:str = self.einTab * self.last_indentation_level


    def __headerTable__(self, position:str = "tbh", add_centering:bool = True)->None:
        """
        # Definition
        Creates the header for a \\table environment in LaTeX. Allows small modifications
        as where should the table be, and if it require to be center the table. This is a 
        'private method' it must not be called outside the class. 

        ## Inputs
            - position: str - Variable to colocate the table (available: t, b, h, or any combination of those, and H)
            - Add_centering:bool - Adds the command '\\centering' to center the tables.

        ## Output
            - None
        
        ## Remarks
        The ''\\scalebox' command is not part of this command, it should be added in the header of the 
        tabular header function, see for example '__headerNAMGM_Table__()'.

        This function adds one indentation due the declaration of the 'table' environment.
        
        """
        #Line rows of the table
        elements_table = [f"\\begin{{table}}[{position}]", "\\centering" if add_centering else ""]
        
        #Indentation levels 
        indentation_levels = [self.last_indentation_level, self.last_indentation_level+1]
        
        #Print the elements in for the header
        for (e,i) in zip(elements_table, indentation_levels): print(self.einTab*i+e)

        #Update the current level of the indentation
        self.last_indentation_level += 1


    def __headerNAMGM_Table__(self, nColumns: int, columns_order:str = "|c|c|c|ccc|c|c|ccc|c|c|ccc|c|"):
        """
        # Definition
        Creates the header for a \\tabular environment in LaTeX. This is completely created to hold
        a specific type of table, which contains 15 columns, and 3 methods, those are NAMGM - {AMG, Queue
        , Random}. 

        ## Inputs
            - nColumns: int - Number of columns in the table (internal information)
            - columns_order: str - Order and position of the columns in the table (positions: l,r,c and possible vlines: |)
        
        ## Output
            - None
        
        ## Remarks
        All header of the tables are different, and must be coded for that type of header. We consider
        that a table has three main components. Header, where we place all the variables; Body, where we
        put the information of those variables; and Footer where we close the table. Each line
        in the table must be see it has a sequence of strings, with an associated indentation.
        That's why we have two variables, one for code rows, and other to specify the indentation of those
        lines.

        This function add's 2 of indentation, one for the 'scalebox' environment and other for the interior
        of the 'tabular' environment.

        """
        #Line rows of the table (descriptive names for an easy manageable)
        tabular_header = [TU.SCALE_BOX_TAB(self.scalebox_value), self.init_tabular(columns_order), TU.T_RULE]
        variable_row_1 = ["\\multirow{2}{*}{$\\bs{F}$} & \\multicolumn{5}{|c}{NAMGM-AMG}  & \\multicolumn{5}{|c}{NAMGM-\\textit{Queue}} & \\multicolumn{5}{|c|}{NAMGM-\\textit{Random}}& \\multirow{2}{*}{Config}"+TU.TAB_END]
        variables_rules = [TU.C_RULE(2,6)+TU.C_RULE(7,11)+TU.C_RULE(12,16)]
        variable_row_2 = ["& C & CT & Iters & Time (s) & Last Norm & C & CT & Iters & Time (s) & Last Norm & C & CT & Iters & Time (s) & Last Norm & "+TU.TAB_END, TU.B_RULE]
        
        #Indentation levels 
        indentation = [self.last_indentation_level, self.last_indentation_level+1]+([self.last_indentation_level+2]*5)

        #All the rows as one sequence of elements
        header = tabular_header + variable_row_1 + variables_rules +variable_row_2        
        
        #Print the elements in for the header
        for (e,i) in zip(header, indentation): print(self.einTab*i + e)

        #Update the current level of indentation
        self.last_indentation_level += 2

    def __footer_Tabular__(self, add_scalebox_end_bracket:bool)->None:
        """
        # Definition    
        Function to print the ending lines of a tabular environment.

        ## Inputs:

            ### Required:

                - add_scalebox_end_bracket: bool - Adds a '{' on a second line to complete the 'scalebox environment'
        
        ## Output:

            - None

        ## Remarks
        
        This is a general footer function for tabular functions. Our design pattern stablish that
        all the ending elements of the table as '\\hlines' or other objects, must stay in the Body
        part of the table. 
        """ 
        #Line rows of the table (descriptive names for an easy manageable)
        end_tabular_part:str = [self.end_tabular, "}" if add_scalebox_end_bracket else ""]

        #Indentation levels 
        indentations = [self.last_indentation_level-1, self.last_indentation_level-2]

        #Print the elements of the footer
        for (e,i) in zip(end_tabular_part, indentations): print(self.einTab*i + e)

        #Decrement the number of indentation
        self.last_indentation_level -=2        



    def __elementTable__(self, information:np.ndarray, deactivate_tab_enders:bool=False):
        """
        # Definition
        This function supposes that somebody give us the results for a factor experiments, and all the methods are flatted in order.
        This is the heart of this script.


        ## Inputs:

            ### Required:
            - `information: np.ndarray`         - A row from the matrix results (check the order of the variables)

            ### Optional:
            - `times_symbol: string = \cdot`     - Multiplication symbol used over the results in scientific notation
            - `add_cell_color: bool = True`      - Boolean flag to add color to the convergency variable.
            - `convergence_color:string = ForestGreen` - Color for the converged methods
            - `not_convergence_color:string = BrickRed` - Color for the not converged methods
            - `cell_alpha: int = 10`             - Transparency of the convergence cells
            - `deactivate_tab_enders: bool = False` - Delete the TAB_END variable at the end of the tables
            - `number_of_decimals: int = 3` - Number of decimals to be displayed on the Last Gradient norm
            - `(tic)Time in scientific: bool = False` - Instead of using decimals points, it's switched to Scientific notation.

        ## Outputs:

            - None - The usage of this function is to print the values of the table

        """ 
        
        #String of the row to be created
        s: str = ""

        #Each variable has its own format
        convergence_flags = [information[i] for i in range(0, information.shape[0], 5)]
        iterations = [information[i] for i in range(1, information.shape[0], 5)]
        convergent_executions = [int(information[i]) for i in range(4, information.shape[0], 5)]
        gradient_norms = [information[i].item() for i in range(3, information.shape[0], 5)]
        
        #We allow the formatting of time if it's required
        if self.time_in_scientific:
            exe_time = [TU.scientific_notation_converter(information[i], self.times_symbol, self.decimals) for i in range(2, information.shape[0], 5)]
        else:
            exe_time = [f"{information[i]: .4f}" for i in range(2, information.shape[0], 5)]
                
        #The creation of the G-Norm is special, so we must process it
        g_formatted = [TU.scientific_notation_converter(g, self.times_symbol, self.decimals) for g in gradient_norms]

        #Creation of the row
        for i in range(0, 3):

            #The convergence is can be partial for the Random - NAMGM (this if handle it)
            if convergence_flags[i] not in [0.0, 1.0]:
                c = f"{convergence_flags[i]: 0.2}" 
                c = f"{{\\scriptsize {c}}}"
                if self.add_colors:
                    c+=TU.construct_cell_color(self.convergence_color, int(convergence_flags[i]*100), self.not_convergence_color, self.convergence_alpha)

            #If the method converged completely or not, then
            else: 

                #Symbols of convergency and not convergency
                if bool(convergence_flags[i]):
                    c =  f"\\textcolor{{{self.convergence_color}}}{{{TU.CHECK_SYMBOL}}}"
                else:
                    if np.isinf(gradient_norms[i]):
                        c = f"\\textcolor{{{self.not_convergence_color}}}{{{TU.CROSS_SYMBOL}}}"
                    else:
                        c = TU.CROSS_SYMBOL


                #Adds color to the convergence cell             
                if self.add_colors:
                    c+= TU.construct_cell_color(self.convergence_color, self.convergence_alpha) if bool(convergence_flags[i]) else TU.construct_cell_color(self.not_convergence_color, self.convergence_alpha)
            

            #We add the number of times that the method end their execution
            c += TU.TAB_LINKER + f"{{\\scriptsize {convergent_executions[i]}}}"

            #Add the other variables
            s += c + TU.TAB_LINKER + f"{iterations[i]:.2f}" + TU.TAB_LINKER + exe_time[i] + TU.TAB_LINKER + g_formatted[i]
            if i!=2:
                s += TU.TAB_LINKER
            else:
                if not deactivate_tab_enders:
                    s += TU.TAB_END
                else:
                    s += ""
        return s

    def __tableBody__(self, factors_matrices: list[np.ndarray], add_config_info:bool = False)->None:
        """ 
        # Definition
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
        
        ## Remark.
        Due that the this is the body of the table; the indentation of the table is in their maximum. It must not
        be increased in this table component.
        
        """
        #Assertions to guarantee the correct working of this function
        assert len(factors_matrices)==3 or factors_matrices.shape[0]==3, "The list of matrices must have the tree configurations [SIMPLEST, ORIGINAL, WithLS]"

        #Current iteration indentation
        indentation:str = (self.last_indentation_level) * self.einTab

        #Print the separation line between factors
        if add_config_info:
            print(indentation +TU.C_RULE(1,15))
            print(indentation + f"\\multirow{{3}}{{*}}{{{self.function_name}}}" + TU.TAB_LINKER + self.__elementTable__(factors_matrices[0], deactivate_tab_enders=True) + TU.TAB_LINKER + "S" +TU.TAB_END)
            print(indentation + TU.TAB_LINKER + self.__elementTable__(factors_matrices[1], True) + TU.TAB_LINKER + "O" +TU.TAB_END)
            print(indentation + TU.TAB_LINKER + self.__elementTable__(factors_matrices[2], True) + TU.TAB_LINKER + "W"+ TU.TAB_END)
            print(indentation + TU.C_RULE(1,17))
        else:        
            #We print the body of the table.
            print(indentation +f"\\multirow{{3}}{{*}}{{{self.function_name}}}" + TU.TAB_LINKER + self.__elementTable__(factors_matrices[0]))
            print(indentation + TU.TAB_LINKER + self.__elementTable__(factors_matrices[1]))
            print(indentation + TU.TAB_LINKER + self.__elementTable__(factors_matrices[2]))
            print(indentation + TU.C_RULE(1,16))


    def __bottomTable__(self, caption:str = "", label:str =""):
        """
        # Definition    
        Function to print the ending lines of a table environment. It allows to place 
        a caption and a label for the current table.

        ## Inputs
            - caption: str - Description of the table
            - label: str - Label to reference the table

        ## Output
            - None

        ## Remarks
        This is a general footer function for table environments.
        """
        #Lines
        footers = [f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    
        #Indentation levels 
        idnttn_levels = [self.last_indentation_level]*2+[self.last_indentation_level-1]

        #Print the elements in the bottom part of the table
        for (e,i) in zip(footers, idnttn_levels): print(self.einTab*i + e) 
        self.last_indentation_level-=1 

        #Verification that the indentation was closed correctly
        assert self.original_indentation == self.last_indentation_level, "There was an error with the indentation variable. It did not ended in their initial place. "


    def generateTable(self, factors_matrices: list[np.ndarray], SGLS_array: np.ndarray, caption:str="", label:str="")->None:
        """
        # Definition.
        Generates the complete concept table for the robust line experiments. 

        ## Inputs:

            ### Required 
            - `factors_matrices: list[np.ndarray]` - A list with the results matrices
            - `SGLS: np.ndarray` - Results matrix for the different factor for the SGLS method
            -
            ### Optional
                - caption: str = "" - Description of the table
                - label: str = ""  - Label to reference the table 
        
        ## Outputs:

            - None - It just prints the table.
        
        ## Remarks.

        """ 
        
        #Init the table
        self.__headerTable__()

        #Print the elements of the NAMGM table results
        self.__headerNAMGM_Table__(15)
        self.__tableBody__(factors_matrices)
        self.__footer_Tabular__(True)

        #Print the table of the SGLS table results
        self.__headerSGLS__Table(5)
        self.__table_SGLS__(SGLS_array)
        self.__footer_Tabular__(True)

        #End the table
        self.__bottomTable__(caption)

    def __headerSGLS__Table(self, nColumns:int = 5, columns_order:str = "|c|c|c|ccc|")->None:
        """
        # Definition
        Creates the header for a \\tabular environment in LaTeX. This is completely created to hold
        a specific type of table, which contains 5 columns, for an unique method, the SGLS.

        ## Inputs
            - nColumns: int - Number of columns in the table (internal information)
            - columns_order: str - Order and position of the columns in the table (positions: l,r,c and possible vertical lines: |)
        
        ## Output
            - None
        
        ## Remarks
        All header of the tables are different, and must be coded for that type of header. We consider
        that a table has tree main components. Header, where we place all the variables; Body, where we
        put the information of those variables; and Footer where we close the table. Each line
        in the table must be see it has a sequence of strings, with an associated indentation.
        That's why we have two variables, one for code rows, and other to specify the indentation of those
        lines.

        This function add's 2 of indentation, one for the 'scalebox' environment and other for the interior
        of the 'tabular' environment.

        """
        tabular_header = [TU.SCALE_BOX_TAB(self.scalebox_value), self.init_tabular(columns_order), TU.T_RULE]
        method_row = [f"\\multicolumn{{{nColumns}}}{{|c|}}{{\\cellcolor{{PerlWhite}}Gradient Descent With LS}}"+TU.TAB_END]
        middle_rule_complete = [TU.C_RULE(1,nColumns)]
        variable_row_1 = ["\\textbf{Problem} & C & CT & Iters & Time (s) & Last Norm"+TU.TAB_END]
        c_rule = [TU.C_RULE(1,5)]

        indentation = [self.last_indentation_level, self.last_indentation_level+1]+([self.last_indentation_level+2]*5)
        header = tabular_header +method_row + middle_rule_complete + variable_row_1 + c_rule
        for (e,i) in zip(header, indentation): print(self.einTab*i + e)
        self.last_indentation_level+=2
        return


    def __table_SGLS__(self, array_of_results: list[np.ndarray])->None:
        """
        # Definition
        This function is a imitation of the __tableBody__, for the SGLS method.
        It does the exact same thing, just instead of dealing with 3 methods, we 
        just work with one. This function was implemented latter, due that the thesis advisor
        ask to add the SGLS in the experiments. 

        ## Inputs

            ### Required 
            - `information: np.ndarray`         - A row from the matrix results (check the order of the variables)

            ### Optional
            - `times_symbol: string = \cdot`     - Multiplication symbol used over the results in scientific notation
            - `add_cell_color: bool = True`      - Boolean flag to add color to the convergency variable.
            - `convergence_color:string = ForestGreen` - Color for the converged methods
            - `not_convergence_color:string = BrickRed` - Color for the not converged methods
            - `cell_alpha: int = 10`             - Transparency of the convergence cells
            - `deactivate_tab_enders: bool = False` - Delete the TAB_END variable at the end of the tables
            - `number_of_decimals: int = 3` - Number of decimals to be displayed on the Last Gradient norm
            - `(tic)Time in scientific: bool = False` - Instead of using decimals points, it's switched to Scientific notation.

        ## Outputs

            - None - The usage of this function is to print the values of the table
        """ 
        #Indentation
        actual_indentation = self.einTab * self.last_indentation_level        
        for i, row in enumerate(array_of_results):

            #Factor 
            s:str = actual_indentation + self.function_name + TU.TAB_LINKER

            #Get the elements of this problem
            c_flag, iters, time, last_norm, convergent_executions  = row
            
            #Formats
            iters = f"{iters: .2f}"
            convergent_executions = f"{int(convergent_executions): d}"
            time = TU.scientific_notation_converter(time, self.times_symbol, self.decimals) if self.time_in_scientific else f"{time: .4f}"
            gnorm = TU.scientific_notation_converter(last_norm, self.times_symbol, self.decimals)

            #The convergence is can be partial for the Random - NAMGM (this if handle it)
            if c_flag not in [0.0, 1.0]:
                c = f"{c_flag: 0.2f}" 
                c = f"{{\\scriptsize {c}}}"
                if self.add_colors:
                    c+=TU.construct_cell_color(self.convergence_color, int(c_flag*100), self.not_convergence_color, self.convergence_alpha)
            #If the method converged completely or not, then
            else: 
                #Symbols of convergency and not convergency
                if bool(c_flag):
                    c =  f"\\textcolor{{{self.convergence_color}}}{{{TU.CHECK_SYMBOL}}}"
                else:
                    if np.isinf(last_norm):
                        c = f"\\textcolor{{{self.not_convergence_color}}}{{{TU.CROSS_SYMBOL}}}"
                    else:
                        c = TU.CROSS_SYMBOL

                #Adds color to the convergence cell             
                if self.add_colors:
                    c+= TU.construct_cell_color(self.convergence_color, self.convergence_alpha) if bool(c_flag) else TU.construct_cell_color(self.not_convergence_color, self.convergence_alpha)

            #Add the number of times that the method converged.
            c += TU.TAB_LINKER + convergent_executions +TU.TAB_LINKER

            #Add the other variables
            s += c + str(iters) + TU.TAB_LINKER + time + TU.TAB_LINKER + gnorm + TU.TAB_END
            print(s)
        #print(actual_indentation+TU.B_RULE)
        return None


def main()->None:

    #Get the values from the parser
    args = parser.parse_args() 

    #Directory of the results original
    problem_path = Path(args.problems_path)
    sgls_table = args.sgls

    #We get all the available problems and we search for the files inside Joint_CSV
    problems = sorted(existing_folders(problem_path))
    #problems_paths = [os.path.join(problem_path,"BRKMCC", "Joined_CSVS")]
    problems_paths = [os.path.join(problem_path, p, "Joined_CSVS") for p in problems]

    #We already know the name of the CSV files inside in the problems_paths. Then for each problem we do
    doomie = latexTable("uwu")
    doomie.__headerTable__()
    doomie.__headerNAMGM_Table__(17)
    i = 0
    for path, problem in zip(problems_paths, problems):
        #We get the dataframes
        data_frames = [pd.read_csv(os.path.join(path, "simplest.csv")), pd.read_csv(os.path.join(path, "original.csv")), pd.read_csv(os.path.join(path, "addedLS.csv"))]
        
        #We process the dataframes
        results_matrix, sgls_mean_array  = create_matrix_results(data_frames)

        #List of matrix results
        list_matrices = results_matrix
        list_arrays_SGLS = [sgls_mean_array]

        #Creation of latexTable object
        table = latexTable(problem, extra_indentation=doomie.last_indentation_level)
     
        #Modification of parameters according to the parser
        table.decimals = 3
        table.time_in_scientific = True

        table.__table_SGLS__(list_arrays_SGLS)

        # if i==0:
        #     table.__tableBody__(list_matrices, add_config_info=True)
        # else:
        #     table.__tableBody__(list_matrices, add_config_info=False) 
        
        i+=1;
    doomie.__footer_Tabular__(True)
    doomie.__bottomTable__()
    return 1

if __name__ =="__main__":
    main()