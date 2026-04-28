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