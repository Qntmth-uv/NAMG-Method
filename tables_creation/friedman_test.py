import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from table_utils import existing_folders

#To manege the information 
import os
from pathlib import Path
import argparse 
import glob

#To visualization
import matplotlib.pyplot as plt
import seaborn as sns



#Parser definition
parser = argparse.ArgumentParser(add_help="Arguments to generate the Friedman test")
parser.add_argument("-p", "--path", help="Path pointing to folders results")
parser.add_argument("-v", "--var", help="Variable to test")
parser.add_argument("--saveplot", action="store_true",  help="Save the bar-plot of the current configuration")


#Group of flags for the set of experiments
group = parser.add_mutually_exclusive_group()
group.add_argument("-c", "--convergency", action="store_true", help="Execute the Friedman test over the convergency experiments.")
group.add_argument("-r2", "--robust2", action="store_true", help="Execute the Friedman test over the Robust experiments ")

#Group of flags for the set of experiments
variables_group = parser.add_mutually_exclusive_group()
variables_group.add_argument("-t", "--time", action="store_true", help="Use the variable 'Execution Time' for the test.")
variables_group.add_argument("-i", "--iterations", action="store_true", help="Use the variable 'Iterations' for the test")

#Group of flags for the configurations
confs_group = parser.add_mutually_exclusive_group()
confs_group.add_argument("-sim", "--simplest", action="store_true", help="Generate the Friedman test for the 'simplest' configuration.")
confs_group.add_argument("-ori", "--original", action="store_true", help="Generate the Friedman test for the 'original' configuration.")
confs_group.add_argument("-add", "--addedLS", action="store_true", help="Generate the Friedman test for the 'withLs' configuration.")
confs_group.add_argument("-all", "--all", action="store_true", help="Generate the Friedman test for all the configurations.")


"""
#Execution commands examples
python3 friedman_test.py -p ../csvs/results/area_robust -r2 -i -sim
python3 friedman_test.py -p ../csvs/results/ -c -i -sim

# To DO.

In the case of all, generate the full data. This is an special type of processing the data.


"""

#CONSTANT VALUES
CONFIGS = ("simplest", "original", "addedLS")
CONVERGENCY  = ("AMG/NAMG", "QUEUE/NAMG", "RANDOM/NAMG", "MD-NEWTON", "BFGS", "SGLS")
ROBUST_II = ("AMG/NAMG", "QUEUE/NAMG", "RANDOM/NAMG", "SGLS") 
COLORS_CHOICED = ['#D55745', '#F6C443', '#5195D4', '#58B89D', '#8448A7', '#7B8BA5', '#64C87A', '#38485C',  "#4E247B"]

#Table Variables Names
METHODS_ORDER = dict(zip(list(range(len(CONVERGENCY))), CONVERGENCY))
PALETTE = dict(zip(CONVERGENCY, COLORS_CHOICED))

def plotting_friedmantest(dataframe_values: pd.DataFrame, variable:str, mode:str, confg:str, palette = None, saveplot:bool = False)->None:
    df_numpy = list(dataframe_values.to_numpy()) 
    statistic, p_value = friedmanchisquare(*df_numpy)
    ranks = dataframe_values.rank(axis=1, ascending=True, method="average")
    mean_rank_value = np.mean(np.mean(ranks))
    join = pd.concat([dataframe_values, ranks], axis=1)
    print(f"ℹ️ Friedman Test {mode}/{variable}/{confg}/: Statistic - {statistic} | P-Value - {p_value}")
    
    #print(np.mean(join.to_numpy(), axis=0))
    #print(join.head(200))
    mean_ranks = ranks.mean().sort_values()

    # 1. Configurar estilo limpio
    sns.set_style("white")
    plt.figure(figsize=(10, 5))

    #Creation of the plot
    if palette is None:
        ax = sns.barplot(
            x=mean_ranks.values, 
            y=mean_ranks.index, 
            hue=mean_ranks.index,
            palette="Spectral",
            edgecolor="none", # Sin bordes negros en las barras
            legend=True,
            #width=0.2 # Espaciado elegante entre barras
        )
    else:
        ax = sns.barplot(
            x=mean_ranks.values, 
            y=mean_ranks.index, 
            hue=mean_ranks.index,
            palette=palette,
            edgecolor="none", # Sin bordes negros en las barras
            legend=False,
            #width=0.2 # Espaciado elegante entre barras
        )

    # 2. Estilo de los Ejes (Grosor y visibilidad)
    # Ocultar bordes superior, derecho e izquierdo para máxima limpieza
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Activar y engrosar solo el eje inferior (X)
    ax.spines['bottom'].set_visible(True)
    #ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')

    # 3. Añadir la línea vertical de umbral (Threshold line)
    plt.axvline(x=mean_rank_value, color='black', linestyle='--', linewidth=1.0, zorder=3)

    # 4. Título y Etiquetas del Gráfico
    plt.title(f'{mode}) Comparative performance in {confg}/{variable}', fontsize=14, fontweight='bold', pad=15, loc='center')
    plt.xlabel('Rank sum', fontsize=8, labelpad=8)
    plt.ylabel('', fontsize=8)

    # 5. Añadir los valores exactos a la derecha de cada barra
    for p in ax.patches:
        val = p.get_width()
        if val > 0: # Evitar errores con valores nulos
            ax.annotate(f'{val:.1f}', 
                        (val, p.get_y() + p.get_height() / 2.), 
                        ha='left', va='center', 
                        xytext=(7, 0), 
                        textcoords='offset points',
                        fontsize=8,
                        fontweight='light')

    plt.tight_layout()
    
    if saveplot:
        saving_path = f"barplot_{mode}_{variable}_{confg}.svg"
        print(f"A file as been created at: {saving_path}")
        plt.savefig(saving_path, format="svg")
    



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

def process_convergency(dataframes: list[pd.DataFrame], variable:str):
    """
    
    ### Remarks. 
    The order of the resulting dataframe is: AMG/NAMGM, QUEUE/NAMGM, RANDOM/NAMGM, NEWTON, BFGS and SGLS. 
    The last two methods (BFGS and SGLS) can be removed to compute a mean over the different configurations. The 
    """
    #Place where we will storage the data
    data = np.zeros((len(dataframes), 30))

    #Obtain the information from the dfs
    for i, df in enumerate(dataframes):
        column = df.columns.to_list()
        data[i, :] = df.to_numpy().flatten()

    #Create the data frame with the information 
    joined_df:pd.DataFrame = pd.DataFrame(data, columns=column * 6)

    #Obtain the values of the variable of interest
    joined_df = joined_df[[variable]]

    #We rename the dataframe with the order of the methods.
    joined_df.columns = CONVERGENCY

    return joined_df

def convergence(path: Path, variable:str):
    paths_to_search = [os.path.join(path, c, "*.csv") for c in CONFIGS]
    configs_dataframes = [get_ListOfDF(p, False)[0] for p in paths_to_search]
    procesed_df = [process_convergency(cdf, variable) for cdf in configs_dataframes] #This contains the DFs for [simplest, original, addedLS]
    
    #We process the BFGS and SGLS
    sgls_dfs = [df[["SGLS"]] for df in procesed_df]
    bfgs_dfs = [df[["BFGS"]] for df in procesed_df]
    SGLS_df = pd.concat(sgls_dfs).groupby(level=0).mean()
    BFGS_df = pd.concat(bfgs_dfs).groupby(level=0).mean()
    
    #Now we remove those two methods
    for i in range(len(procesed_df)):
        procesed_df[i] = procesed_df[i].iloc[:, 0:4]
    
    #Finally we append the mean dfs (BFGS and SGLS) and we return the DF per conf
    final_dfs = [pd.concat([c, BFGS_df, SGLS_df], axis=1) for c in procesed_df] 
    df_per_conf = dict(zip(CONFIGS, final_dfs)) 
    
    return df_per_conf

def process_Robust_II(conf_dataframes: list[pd.DataFrame], problem_names:str):
    """
    ROBUST II

    1. Simplest
    2. Original
    3. AddedLS
    4. SGLS

    ### Inputs:
        -conf_dataframes(list[pd.DataFrames]): List of problems results in the configuration.
    ### Outputs:
        -df(pandas.DataFrame): Bidimensional dataframe with rows as problems and columns as variables

    ### Remarks.
    The order in the final dataframe is given by the CSVS order. This is a fix order, therefore the order
    of the dataframe is AMG-NAMGM, QUEUE-NAMGM, RANDOM-NAMGM, SGLS.
    """
    matrix_results = np.zeros((len(conf_dataframes), 20))
    for i, df in enumerate(conf_dataframes):
        data = df.to_numpy()
        matrix_results[i, :] = data.flatten()
    col_names = conf_dataframes[0].columns.to_list()
    conf_df = pd.DataFrame(matrix_results, index=problem_names,columns=col_names*4)
    return conf_df

def robust_II(path: Path, problems: list[str], variable:str = "Iterations"):
    """
    
    ### Inputs:
        -Required

        - Optional.
            - variable(str): Variable to measure (Iterations, Execution Time, Last Gradient, Percentage of convergence, Not divergency number)
    
    ### Outputs:
        - conf_dicts(dict): A dictionary with keys the configurations and as values their processed DF.
    """
    #We will search inside the area_robust/problem/Joined_CSVS
    paths_to_search = [os.path.join(path, p, "Joined_CSVS", "*.csv") for p in problems]
    
    #We read the problems in al the area_robust folder. The result is a List of N list of 3 dataframes (simplest, original, addedLS)    
    configs = [get_ListOfDF(p, True)[0] for p in paths_to_search]

    #Get the information according to the configuration
    simplest_dfs = [c[0] for c in configs]
    original_dfs = [c[1] for c in configs]
    withLS_dfs = [c[2] for c in configs]

    configs = [simplest_dfs, original_dfs, withLS_dfs]

    #This is the function that makes all the magic.
    arrays_problms = [process_Robust_II(c, problems) for c in configs]

    #Recolection of the information of SGLS into a single dataframe
    sgls_dfs = [df_conf.iloc[:, 15:] for df_conf in arrays_problms]
    mean_df = pd.concat(sgls_dfs).groupby(level=0).mean()

    #Now we can delete those columns where the SGLS appears. And we finally get the NAMGMS configuations
    variables_dfs = [c.iloc[:,0:15][variable] for c in arrays_problms]
    v_sgls_df = mean_df[variable]

    #Columns_names and cration of the final df per conf. 
    methods_names = ["AMG/NAMG", "QUEUE/NAMG", "RANDOM/NAMG", "SGLS"] 
    final_dfs =  [pd.concat([c, v_sgls_df], axis=1) for c in variables_dfs]
    for c in final_dfs:
        c.columns = methods_names
    df_per_conf = dict(zip(CONFIGS, final_dfs))
    return df_per_conf



def process_all(dict_of_confs: dict, is_r2:bool = False):


    #Creation of the final dataframe
    list_df = list(dict_of_confs.values())
    

    #Creation of the columns names for the complete dataframe
    acronyms = ["S", "O", "W"]    
    convergency_acronyms = ["A", "Q", "R", "MN"]
    new_columns = []
    if is_r2:
        columns_names = ROBUST_II[:-1]
        for conf in acronyms:
            for method in columns_names:
                new_columns.append(conf + "-" + method)
        new_columns.append(ROBUST_II[-1])

        #Extract the SGLS column and remove it from the other dfs
        sgls = list_df[0][["SGLS"]]
        for i in range(len(list_df)):
            list_df[i] = list_df[i].iloc[:, :-1]
        list_df.append(sgls)
    else:
        columns_names = CONVERGENCY[0:4]
        for conf in acronyms:
            for method in convergency_acronyms:
                new_columns.append(conf + "-" + method)
        new_columns+=CONVERGENCY[4:]

        #Extract the BFGS and SGLS columns and we remove it from the other dfs
        sgls = list_df[0][["SGLS"]]
        bfgs = list_df[0][["BFGS"]]
        for i in range(len(list_df)):
            list_df[i] = list_df[i].iloc[:, :-2]
        list_df.append(sgls)
        list_df.append(bfgs)

    #Create the final dataframe
    all_methods = pd.concat(list_df, axis=1)
    all_methods.columns = new_columns

    return all_methods, new_columns

def main():
    #ARGS
    args = parser.parse_args()
    path = Path(args.path)

    #Variables arguments
    time = args.time
    iters = args.iterations
    save_plot = args.saveplot

    #Experiments arguments
    convergency = args.convergency
    r2 = args.robust2

    #Configuration arguments
    flag_simplest = args.simplest    
    flag_orignal = args.original
    flag_addedLS = args.addedLS
    flag_all = args.all
    print("Reading from path:", path)
    

    #Variable to execute the test    
    if time:
        VAR = "Execution time"
        VAR_name = "TIME"
    elif iters:
        VAR = "iterations"
        if r2:
            VAR = "Iterations"
        VAR_name = "Iters"
    else:
        return -1

    #Experiments sets creation
    if convergency:
        EXP = "CONV"
        final_table = convergence(path, VAR)
    elif r2:
        EXP = "ROBUSTII"
        problems = existing_folders(path=path)
        final_table = robust_II(path, problems, VAR)
    else:
        return -1

    #Configuration to use (the final table contains a dict with key the conf and value their df)
    if flag_simplest:
        CONF = "simplest"
    elif flag_orignal:
        CONF = "original"
    elif flag_addedLS:
        CONF = "addedLS"
    elif flag_all:
        CONF = "All-Methods"
    else:
        return -1             

    #We have to manage the concatenation of the resulting DFs
    if CONF in ["simplest", "original", "addedLS"]:
        #Creation of plot of the Friedman test
        data = final_table[CONF]
        plotting_friedmantest(data, VAR_name, EXP, CONF, palette = PALETTE, saveplot=save_plot)
    else:
        #Creation of plot of the Friedman test
        data = final_table 

        #Convergency
        if not r2:
            data, columns_name = process_all(data, False)
            colors_choiced = ['#D55745', '#F6C443', '#5195D4', '#58B89D']*3 + ['#38485C',  "#4E247B"]
            palette = dict(zip(columns_name, colors_choiced))
        #Robust Noise
        else:
            data, columns_name = process_all(data, True)
            colors_choiced = ['#D55745', '#F6C443', '#5195D4']*3 + ["#4E247B"]
            palette = dict(zip(columns_name, colors_choiced))
        
        plotting_friedmantest(data, VAR_name, EXP, CONF, palette, saveplot=save_plot)
        
    return 0

if __name__ == "__main__":
    main()    
    