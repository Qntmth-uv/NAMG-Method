import math
import numpy as np

#Table constants 
TAB_LINKER = " & "
TAB_END = "\\\\"
B_RULE = "\\bottomrule"
M_RULE = "\\midrule"
T_RULE = "\\toprule"
C_RULE = lambda l,r: "\\cmidrule(lr){"+str(l)+"-"+str(r)+"}"
SCALE_BOX_TAB = lambda s: f"\\scalebox{{{s}}}"+"{"

#Multi-columns/rows command generators
MULTI_COL = lambda columns_to_occupy, position, content: f"\\multicolumn{{{columns_to_occupy}}}{{{position}}}{{{content}}}"
MULTI_ROW = lambda rows_to_occupy, content: f"\\multirow{{{rows_to_occupy}}}{{*}}{{{content}}}"

#Symbols
CHECK_SYMBOL = "\\checkmark"
CROSS_SYMBOL = "\\texttimes"

#Text modifiers
TEXT_C = lambda color, content: f"\\textcolor{{{color}}}{{{content}}}" #Text color command shortcut

#Table rules modifiers
T_RULE = "\\toprule"
B_RULE = "\\bottomrule"
M_RULE = "\\midrule"
C_RULE = lambda l,r: "\\cmidrule(lr){"+f"{l}-{r}"+"}"
SCALE_BOX_TAB = lambda val: f"\\scalebox{{{val}}}" + "{"

def construct_cell_color(color1:str, alpha1:int, color2:str =None, alpha2:int=None)->str:
    """
    # Definition
    
    Function to construct the color cell on a LaTeX table. If color2 and alpha 2 are
    not provided, it still works.
    
    ## Input:
        - `color1`: string - First (or main) color of the cell
        - `alpha1`:   int  - Transparency (or percentage of the first color) 
        - `color1`(optional): string - Second color of the cell
        - `alpha2`(optional):   int  - Transparency of the combined color 

    ## Output:
        - `s`: string - LaTeX command to color the cell
    """
    final_color:str = f"{color1}!{alpha1}"
    if color2:
        final_color += f"!{color2}"
        if alpha2:
            final_color += f"!{alpha2}"
    return f"\\cellcolor{{{final_color}}}"


def scientific_notation_converter(G:float, times_symbol:str = "\\cdot", number_of_decimals:int = 3, remove_math_mode:bool = False)->str:
    """
    # Usage
    Given a number G, is written in the scientific notation for LaTeX. In other words, in the format 'M \\multiplication_symbol Exponent'.

    ## Inputs.
        ## Required.
            - G(float): Real number or infinite.

        ## Optional.
            - times_symbol(str) = '\\cdot': Symbol to represent multiplication 
            - number_of_decimals(int) = 3: Decimal numbers after the dot.
            - remove_math_mode(bool) = False: Removes the '$$' over the constructed representation.

    ## Output.
        - s(str): Representation of the number in scientific notation for LaTeX usage.
    
    """
    #String to be constructed
    s:str = ""
    
    #If the number is infinite
    if np.isinf(G):
        s = "$\infty$"
    
    #If the number is exactly zero
    elif G == 0.0:
        s= "0"

    #The number is not a extreme case
    else:
        #If the number it is not a infinite number or zero we can apply the log
        exponent = int(math.floor(math.log10(abs(G))))
        decimal = G / (10 ** exponent)

        #Format depending of the number of desired decimals
        if number_of_decimals==4:
            s = f"{decimal: .4f}{times_symbol} 10^{{{int(exponent)}}}"
        elif number_of_decimals ==5:
            s = f"{decimal: .5f}{times_symbol} 10^{{{int(exponent)}}}"
        else:
            s = f"{decimal: .3f}{times_symbol} 10^{{{int(exponent)}}}"
    return f"${s}$" if not remove_math_mode else s


def add_bold_to_element(element:any, math_mode:bool, math_bold_command:str = '\\mathbf',
                        is_best_value:bool = False, color_best:str = 'ForestGreen', add_math_mode:bool = True,
                        it_converged: bool = True)->str:
    """
    # Usage.
    Allows the addition of bold to a value, it does not matter if the value is a string or a float value. We can 
    choice how to add the bold to that value. Also this function allows to color the bolded value, given the color. This allows 
    to highlight even between best values. 

    ## Input:

        ### Required:

        - element(str | float): Value to add bold. 
        - math_mode(bool): Changes the command to add bold. In math mode use the `math_bold_command` otherwise it uses `\\textbf`    

        ### Optional;
        
        - math_bold_command(str) = '\\mathbf{}': Command to add bold to value (can also used \\mathbf{}')
        - is_bets_value(bool) = False: Adds color to the highlight the value.
        - color_best(str)= 'ForestGreen': Add the color to the 'best_value'.
        - add_math_mode(bool) = True: Adds '$$' to the bolded value.
        - it_converged(bool) = True: If the method does not converge, then we dont allow the addition of bold. Even if it is the best value.
    
    ## Outputs:
        - s(str): Bolded value

    ## Remarks.
    We dont allow the bolding of a value if it does not converge, due to that it is not relevant.
    """
    #Variable where we will put the bolded string
    bolded_element: str = ""

    if it_converged:
        #Math mode
        if math_mode:
            bolded_element = math_bold_command + f"{{{element}}}" 
            if add_math_mode: bolded_element = f"${bolded_element}$"

        #Normal mode        
        else:
            bolded_element = f"\\textbf{{{element}}}"

        #If its required to add the color    
        bolded_element = TEXT_C(color_best, bolded_element) if is_best_value else bolded_element

        return bolded_element
    
    #If the method does not converged, then do not add the bold.
    else:
        return element
    
def values_and_index_best(data: np.ndarray, order: list):
    """
    ## Usage.
    
    Function to obtain the best values of the configurations. It uses the corresponding order
    and given data
    
    ### Returns:
    - `best_per_conf(np.ndarray)`: Best values in each configuration and variable
    - `index_best_conf(np.ndarray)`: Index where are located such best values
    - `best_global(np.ndarray)`: Best values per variable across all the configurations
    - `index_best_global(np.ndarray)`: Index of the best global results
    """
    # Where to save the best values and index 
    max_per_variable = []
    index_maxPV = []
    global_val = []
    index_global_val = []

    #Iteration to get the values
    for v in order:
        variable = data[:, v, :]
        evaluator = np.max if v==3 else np.min
        evaluator_idx = np.argmax if v==3 else np.argmin
        max_values = []
        index = []
        for (c, result) in enumerate(variable):
            max_values.append(evaluator(result).item())
            index.append([c, evaluator_idx(result).item(), v])
        max_per_variable.append(max_values)
        index_maxPV.append(index)
        global_val.append(evaluator(max_values).item())
        index_global_val.append(index[evaluator_idx(max_values).item()])

    #Convert into numpy arrays
    max_per_variable = np.array(max_per_variable).reshape(-1, 3).tolist()
    index_maxPV = np.array(index_maxPV).reshape(-1, 3).tolist()
    global_val = np.array(global_val).reshape(-1).tolist()
    index_global_val = index_global_val

    return max_per_variable, index_maxPV, global_val, index_global_val


class LaTeXTable:
    def __init__(self, scalebox:float = 1.0, colors:bool = True, extra_indentation:int = 0):
        #Assertion of values
        assert 0.0 <= scalebox <= 1, "The scalebox_value value must be on the interval [0, 1]"

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

        #Table objects
        ##Top
        self.position:str = "tbh"
        self.add_centering:bool = True

        ##Bottom
        self.caption:str = ""
        self.label:str = ""
         
        
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
        #Save the given configuration
        self.position = position
        self.add_centering = add_centering

        #Line rows of the table
        elements_table = [f"\\begin{{table}}[{position}]", "\\centering" if add_centering else ""]
        
        #Indentation levels 
        indentation_levels = [self.last_indentation_level, self.last_indentation_level+1]
        
        #Print the elements in for the header
        for (e,i) in zip(elements_table, indentation_levels): print(self.einTab*i+e)

        #Update the current level of the indentation
        self.last_indentation_level += 1


    def __bottomTable__(self, caption:str = "", label:str =""):
        """
        # Definition.

        Function to print the ending lines of a table environment. It allows to place 
        a caption and a label for the current table.

        ## Inputs:

            ### Optional.
            - caption(str) = '': Table description.
            - label(str) = '': Table's label.

        ## Outputs:
            - None

        ## Remarks
        This is a general footer function for table environments.
        """

        #Save the values from the bottom of the table
        self.caption:str = caption
        self.label:str = label

        #Lines
        footers = [f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    
        #Indentation levels 
        idnttn_levels = [self.last_indentation_level]*2+[self.last_indentation_level-1]

        #Print the elements in the bottom part of the table
        for (e,i) in zip(footers, idnttn_levels): print(self.einTab*i + e) 
        self.last_indentation_level-=1       

        #Verification that the indentation was closed correctly
        assert self.original_indentation == self.last_indentation_level, "There was an error with the indentation variable. It did not ended in their initial place. "

    def generateTable(self, tables: list[callable], **kwargs):
        self.__headerTable__()
        for i in range(len(tables)):
            tables[i].generateTabular()
        self.__bottomTable__()


#Generic latex Tabular environment. This to be used as a basis of other 
class genericLaTeXTabular:
    def __init__(self):
        pass



def main()->int:
    table = LaTeXTable(0.1, True, 0)
    
    
    #table.generateTable([])
    return 0

if __name__ == "__main__":
    main()