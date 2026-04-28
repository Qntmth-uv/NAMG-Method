import math
import numpy as np

#Table constants 
TAB_LINKER = " & "
TAB_END = "\\\\"
CHECK_SYMBOL = "\\checkmark"
CROSS_SYMBOL = "\\texttimes"

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


def scientific_notation_converter(g:float, times_symbol:str = "\\cdot", number_of_decimals:int = 3)->str:
    """jdwjwj"""
    #String to be constructed
    s:str = ""
    
    #If the number is infinite
    if np.isinf(g):
        s = "$\infty$"
    
    #If the number is exactly zero
    elif g == 0.0:
        s= "0"

    #The number is not a extreme case
    else:
        #If the number it is not a infinite number or zero we can apply the log
        exponent = int(math.floor(math.log10(abs(g))))
        decimal = g / (10 ** exponent)

        #Format depending of the number of desired decimals
        if number_of_decimals==4:
            s = f"${decimal: .4f}{times_symbol} 10^{{{int(exponent)}}} $"
        elif number_of_decimals ==5:
            s = f"${decimal: .5f}{times_symbol} 10^{{{int(exponent)}}} $"
        else:
            s = f"${decimal: .3f}{times_symbol} 10^{{{int(exponent)}}} $"
    return s