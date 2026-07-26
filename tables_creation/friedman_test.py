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
from matplotlib.colors import LogNorm
import seaborn as sns

#Parser definition
parser = argparse.ArgumentParser(add_help="Arguments to generate the Friedman test")
parser.add_argument("-p", "--path", help="Path pointing to folders results")
parser.add_argument("-v", "--var", help="Variable to test")
parser.add_argument("--saveplot", action="store_true",  help="Save the bar-plot of the current configuration")

#Group of flags for the set of experiments
group = parser.add_mutually_exclusive_group()
group.add_argument("-c", "--convergency", action="store_true", help="Execute the Friedman test over the convergency experiments.")
group.add_argument("-r1", "--robust1", action="store_true", help="Execute the Friedman test over the Robust I experiments ")
group.add_argument("-r2", "--robust2", action="store_true", help="Execute the Friedman test over the Robust II experiments ")

#Group of flags for the set of experiments
variables_group = parser.add_mutually_exclusive_group()
variables_group.add_argument("-t", "--time", action="store_true", help="Use the variable 'Execution Time' for the test.")
variables_group.add_argument("-i", "--iterations", action="store_true", help="Use the variable 'Iterations' for the test")
variables_group.add_argument("-cf", "--convergence_flag", action="store_true", help="Use the convergence flag to measure the percentage of convergence (Robust-II)")
variables_group.add_argument("-its", "--iters_per_seconds", action="store_true", help="Use the variable Iterations/Sec (Only for Convergency)")

#Group of flags for the configurations
confs_group = parser.add_mutually_exclusive_group()
confs_group.add_argument("-sim", "--simplest", action="store_true", help="Generate the Friedman test for the 'simplest' configuration.")
confs_group.add_argument("-ori", "--original", action="store_true", help="Generate the Friedman test for the 'original' configuration.")
confs_group.add_argument("-add", "--addedLS", action="store_true", help="Generate the Friedman test for the 'withLs' configuration.")
confs_group.add_argument("-all", "--all", action="store_true", help="Generate the Friedman test for all the configurations.")

"""
jxecution commands examples
python3 friedman_test.py -p ../csvs/results/area_robust -r2 -i -sim
python3 friedman_test.py -p ../csvs/results/ -c -i -sim --saveplot
python3 friedman_test.py -p ../csvs/results/ -c -i -ori --saveplot
python3 friedman_test.py -p ../csvs/results/ -c -i -add --saveplot
python3 friedman_test.py -p ../csvs/results -c -i -all --saveplot

#Results of It/Sec 
python3 friedman_test.py -p ../csvs/results -c -its -all 

#Problems related to R1
python3 friedman_test.py -p ../csvs/results/robust -r1 -i -add 

#Execution related to the showed results in section Robust II(convergence flag)
python3 friedman_test.py -p ../csvs/results/area_robust -r2 -cf -all 

"""

#CONSTANT VALUES
CONFIGS = ("simplest", "original", "addedLS")
CONVERGENCY  = ("AMG", "QUEUE", "RANDOM", "MD-NEWTON", "BFGS", "GDLS")
ROBUST_I = ("GDLS", "AMG", "QUEUE", "RANDOM") 
ROBUST_II = ("AMG", "QUEUE", "RANDOM", "GDLS") 
COLORS_CHOICED = ['#D55745', '#F6C443', '#5195D4', '#58B89D', '#8448A7', '#7B8BA5', '#64C87A', '#38485C',  "#4E247B"]
PROBLEM_DIM_DICT = {
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

#Table Variables Names
METHODS_ORDER = dict(zip(list(range(len(CONVERGENCY))), CONVERGENCY))
PALETTE = dict(zip(CONVERGENCY, COLORS_CHOICED))

def plotting_friedmantest(dataframe_values: pd.DataFrame, variable:str, mode:str, confg:str, palette_colors= None, saveplot:bool = False,
                          less_is_better:bool = True, plot_conf:dict = None)->None:
    df_numpy = list(dataframe_values.to_numpy()) 
    statistic, p_value = friedmanchisquare(*df_numpy)
    ranks = dataframe_values.rank(axis=1, ascending=less_is_better, method="average")
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
    if palette_colors is None:
        ax = sns.barplot(
            x=mean_ranks.values, 
            y=mean_ranks.index, 
            hue=mean_ranks.index,
            palette="Spectral",
            edgecolor="none",
            legend=True,
            #width=0.2
        )
    else:
        ax = sns.barplot(x=mean_ranks.values, y=mean_ranks.index, hue=mean_ranks.index, palette=palette_colors, edgecolor="none", legend=False)

    # Ocultar bordes superior, derecho e izquierdo para máxima limpieza
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Activar y engrosar solo el eje inferior (X)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    
    # 3. Añadir la línea vertical de umbral (Threshold line)
    plt.axvline(x=mean_rank_value, color='black', linestyle='--', linewidth=1.0, zorder=3)

    # 4. Título y Etiquetas del Gráfico
    plt.title(f"{plot_conf["EXP"]}: {plot_conf["VAR"]} across {plot_conf["CONF"]}", fontsize=14, fontweight='bold', loc = "center")
    #plt.title(f"{plot_conf["EXP"]}: {plot_conf["VAR"]} across {plot_conf["CONF"]}", fontsize=14, fontweight='bold', pad=15, loc='center')
    plt.xlabel('Ranked sum', fontsize=8, labelpad=8)
    plt.ylabel('', fontsize=8)

    # 5. Añadir los valores exactos a la derecha de cada barra
    for p in ax.patches:
        val = p.get_width()
        if val > 0: # Evitar errores con valores nulos
            ax.annotate(f'{val:.1f}', (val, p.get_y() + p.get_height() / 2.),  ha='left', va='center', xytext=(7, 0), textcoords='offset points',
                        fontsize=8, fontweight='light')

    plt.tight_layout()
    
    if saveplot:
        #Modification of the principal variables to allow the correct saving of a plot
        mode = mode.replace(" ", "_")
        variable = variable.replace(" ", "_")
        confg = confg.replace(" ", "_")

        #Saving the plot and sharing information
        saving_path = f"barplot_{mode}_{variable}_{confg}.svg"
        print(f"A file as been created at: {saving_path}")
        plt.savefig(saving_path, format="svg")
    else:
        plt.show()
        plt.clf()


def plotting_heat_map(data: pd.DataFrame, saveplot:bool = False, anottations:bool = False, experiment:str = "R1", use_log_scale:bool = False):
    #Size of the plot    
    plt.figure(figsize=(10, 8))

    """
    TODO: Keep constant the name of the methods across all different experiments. In this moment we use the 
    following if to keep it usage and to make the desired plot, nevertheless this is a minor change 
    that can be useful to the main thesis PDF.

    by:   Qntmth-UV
    date: July 15 26    
    """

    #Here we modify the order of the plot to show from the worts 
    if(experiment == "C"):
        desired_order = ["S-R", "GDLS", "O-R", "S-MN", "O-MN", "W-R", "BFGS", "W-MN", "S-Q", "W-Q", "O-Q", "S-A", "O-A", "W-A"]
        data = data[desired_order]        
    elif(experiment == "R1"):
        desired_order = ["GDLS", "O-Q", "W-R", "O-R", "O-A", "S-R", "W-Q", "W-A", "S-Q", "S-A"]
        data = data[desired_order]

    elif(experiment == "R2"):
        desired_order = ["GDLS", "O-Q", "O-A", "O-R", "W-R", "S-Q", "W-A", "S-R", "W-Q", "S-A"]
        data = data[desired_order]
    else:
        data = data
        

    #Plot configuration
    if use_log_scale:
        ax = sns.heatmap(data, cmap='viridis', annot=anottations, fmt=".1f", annot_kws={"color": "black", "fontsize": 6}, norm=LogNorm())
    else:
        ax = sns.heatmap(data, cmap='RdYlGn', annot=anottations, fmt=".1f", annot_kws={"color": "black", "fontsize": 6})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left')

    #Save the plot for usage
    if saveplot:
        saving_path = f"heatMap_{experiment}_CA_ALL.svg"
        print(f"A file as been created at: {saving_path}")
        plt.savefig(saving_path, format="svg", bbox_inches="tight", pad_inches=0.1)

    #Only visualization of the heatmap (before of saving it)
    else:
        plt.show()
        plt.clf()

def plotting_evolution(information: pd.DataFrame, palette: list[str], saveplot:bool, **kwargs):
    #Variables
    # var_name = kwargs["VAR"]
    # conf_name = kwargs["CONF"]

    #Addition of the dimension of the problem using the index
    modified_info = information.copy()
    modified_info['Dimension'] = modified_info.index.map(PROBLEM_DIM_DICT)

    #Clean of the data-frame to avoid the not-valid items due to not convergence
    modified_info.replace(0, 1000.03, inplace=True)

    #We group by dimensiona and we compute the mean
    modified_info = modified_info.groupby('Dimension').mean()

    #Plot and modifications of the it 
    ax = modified_info.plot(kind='line', color = palette, figsize=(10, 6), alpha=0.5)
    ax.legend(palette)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, "major", "both", alpha=0.2)
    plt.xlabel('Problem Dimension')
    plt.ylabel('It/Sec')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.title('Convergence: Iterations per seconds acrross all configurations')
    plt.tight_layout()

    #Save the plot for usage
    if saveplot:
        saving_path = "evolution_ITS_ALL.svg"
        print(f"A file as been created at: {saving_path}")
        plt.savefig(saving_path, format="svg", bbox_inches="tight", pad_inches=0.1)

    #Only visualization of the heatmap (before of saving it)
    else:
        plt.show()
        plt.clf()


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
    #Paths to search per configuration
    paths_to_search = [os.path.join(path, c, "*.csv") for c in CONFIGS]

    #List of list with dataframes per configuration
    read_df_informations = [get_ListOfDF(p, False) for p in paths_to_search]
    configs_dataframes = [data[0] for data in read_df_informations] #DATA
    problems_in_df = [data[1] for data in read_df_informations]     #Problems data
    
    #Assertion that all the configurations have the same number of problems in the same order. 
    assert len(problems_in_df) == 3, "The list of problems is not complete" #Correct reading of the problems per configuration
    problems_simplest, problems_original, problems_withls = problems_in_df

    #Assertion that all list have the same problems in the same order
    assert len(problems_simplest) == len(problems_original) and len(problems_original) == len(problems_withls), f"The list of problems doesn't have the same length: S: {len(problems_simplest)}, O: {len(problems_original)}, W: {len(problems_withls)}"
    for i, (p_s, p_o, p_w) in enumerate(zip(problems_simplest, problems_original, problems_withls)):
        assert (p_s == p_o) and (p_o == p_w), f"There is an error in the item ({i}), the next problems does not match: S: {p_s}, O: {p_o}, W: {p_w}"

    #We save all the problems in just one variable (we can't remove the data)
    problems = problems_simplest = problems_original = problems_withls
    print("The reading was correct and the order of the problems are equal, process continued without problem.")

    #We process the list of dataframes per configuration
    procesed_df = [process_convergency(cdf, variable) for cdf in configs_dataframes] #This contains the DFs for [simplest, original, addedLS]
    
    #We process the BFGS and GDLS
    sgls_dfs = [df[["GDLS"]] for df in procesed_df]
    bfgs_dfs = [df[["BFGS"]] for df in procesed_df]
    SGLS_df = pd.concat(sgls_dfs).groupby(level=0).mean()
    BFGS_df = pd.concat(bfgs_dfs).groupby(level=0).mean()
    
    #Now we remove those two methods
    for i in range(len(procesed_df)): procesed_df[i] = procesed_df[i].iloc[:, 0:4]
    
    #Finally we append the mean dfs (BFGS and SGLS) and we return the DF per conf
    final_dfs = [pd.concat([c, BFGS_df, SGLS_df], axis=1) for c in procesed_df] 
    df_per_conf = dict(zip(CONFIGS, final_dfs)) 
    
    return df_per_conf, problems

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
    methods_names = ["AMG", "QUEUE", "RANDOM", "GDLS"] 
    final_dfs = [pd.concat([c, v_sgls_df], axis=1) for c in variables_dfs]
    for c in final_dfs: c.columns = methods_names
    df_per_conf = dict(zip(CONFIGS, final_dfs))
    return df_per_conf
    
def robust_I(path: Path, variable:str = "Convergence_flag"):

    #Check for each configuracion
    conf_paths = [Path(os.path.join(path, c)) for c in CONFIGS]
    problems_in_conf = [existing_folders(cp) for cp in conf_paths]

    #We sort the problems in each configuration to check if there is an error reading 
    for p in problems_in_conf: p.sort()

    #All configurations have the same problems and size
    assert len(problems_in_conf[0]) == len(problems_in_conf[1]) and len(problems_in_conf[1]) == len(problems_in_conf[2]), "The number of problems in the configurtions is different. Please verify this"
    for problm_sim, problm_ori, problm_add in zip(*problems_in_conf):
        assert problm_sim == problm_ori and problm_ori == problm_add, f"There is a difference between the readed problems: S:{problm_sim} - O:{problm_ori} - W:{problm_add}"

    #Now that we assert that all conf has the same amount of problems, we dont require to save all the problems
    problems_in_conf = problems_in_conf[0]

    #Now per configuration create a new dictionary to hold the results matrix per problem; eg. simplest_dc[WOODS] = MATRIX_WOODS
    simplest_dc, original_dc, added_dc = dict(), dict(), dict() #Initialization of Hash Maps
    configs_dic = dict(zip(CONFIGS, [simplest_dc, original_dc, added_dc]))

    #We read all the problems with their factors and we process them 
    for p in problems_in_conf:
        for (i, c) in enumerate(CONFIGS):
            
            #Path where we will search
            path_of_factors = os.path.join(path, c, p, "*.csv")

            #Read all the factors in the path where we search
            list_dfs = get_ListOfDF(path_of_factors)[0]

            #Process the data into a dataframe
            data_problem = __process_csv__(list_dfs)

            #Create and entry for the problem in the current configuraiton
            configs_dic[c][p] = data_problem
    
    #Verificaytion that all the problems in the correspondent entry have the same dimensions
    for p in problems_in_conf:
        assert configs_dic["simplest"][p].shape == configs_dic["original"][p].shape and configs_dic["original"][p].shape == configs_dic["addedLS"][p].shape, "The dimensions of the dataframes are not equal"

    #Now that we have verified that all problems have the same dimension of information we take the information that we requiere.
    final_df = process_Robust_I(configs_dic, problems_in_conf) #This returns all the in just one dataframe

    """
    TODO: SEPARATE THE DATAFRAMES PER CONFIG AND CREATE A DICT TO KEEP THE STRUCTURE OF THIS CODE.

    Althouht this is important, right now is not primordial. 

    Att. Qmtmth-UV
    Date: Juli 10 26
    """

    return final_df


def process_Robust_I(config_dic: dict, problems_keys: list[str], variable:str = "Convergence_percentage"):
    """
    ### Remark.
    For this experiment we require to check if the method converges. Known how much for each one
    the refore the variable that we measure is the 'Convergence_percentage' used in the process_csv
    function.
    """
    #After we sum the values for each problem in the list and we the dataframe with the values
    if variable == "Convergence_percentage":
        sumed_values = np.zeros(shape=(len(problems_keys), 10))
        for (i, p) in enumerate(problems_keys):

            #Get the information and 
            problems_df = [config_dic[c] for c in CONFIGS]
            variable_to_work = [df[p][variable] for df in problems_df] #3 df with 4 entries each
            non_repeated_methods = [vk.iloc[:,1:] for vk in variable_to_work] #Removed GDLS
            GDLS_dfs = [vk.iloc[:, 0] for vk in variable_to_work] # Only GDLS
            mean_df = pd.concat(GDLS_dfs).groupby(level=0).mean().sum() 
            variable_to_work = [vk.sum(axis=0).to_numpy() for vk in non_repeated_methods]

            #Fill the values on the data frame
            sumed_values[i, 0] = mean_df #GDLS
            for j in range(3): sumed_values[i, 1 + 3*j:3*j + 4] = variable_to_work[j]

        #Creation of the final data frame with the results per configuration
        data = pd.DataFrame(sumed_values)
        data.columns = ["GDLS", "S-A", "S-Q", "S-R", "O-A", "O-Q", "O-R", "W-A", "W-Q", "W-R"]
        data.index = problems_keys
        return data
    else:
        raise ValueError("Incorrect variable to work with")

def __process_csv__(dataframes_list: list[pd.DataFrame]):
    #Name of the columns of the dataframe
    """
    1. Convergence 
    2. Number of iterations taken
    3. Execution time
    4. Last gradient norm of the sequence
    5. Number of divergences"""
    names = ["Convergence_percentage", "iterations", "time", "norm", "number_of_div"] * 4

    # We have 3 methods (N) and 5 variables, then in total we have N \times 5 elements in the dataframe
    result_matrix = np.zeros((len(dataframes_list), 5*4)) #(Number of factors, Variables (5 variables * 3 methods))

    #Picking of the information of each method (picking in rows)
    for i in range(0, len(dataframes_list)):
        results_variables_per_factor: np.ndarray = dataframes_list[i].to_numpy()
        result_matrix[i,:] = results_variables_per_factor.flatten()

    df = pd.DataFrame(result_matrix)
    df.columns = names
    return df



def process_all(dict_of_confs: dict, is_r2:bool = False):
    """Function to process the 9 NAMGM-Methods, Newtons Mod, SGLS and BFGS"""
    #Creation of the final dataframe
    list_df = list(dict_of_confs.values())

    #Creation of the columns names for the complete dataframe
    acronyms = ["S", "O", "W"]    
    robust_II__acronyms = ["A", "Q", "R", "GDLS"]
    convergency_acronyms = ["A", "Q", "R", "MN"]
    new_columns = []
    
    #If the experiment is Robust II
    if is_r2:
        columns_names = convergency_acronyms[:-1]
        for conf in acronyms:
            for method in columns_names:
                new_columns.append(conf + "-" + method)
        new_columns.append(ROBUST_II[-1])

        #Extract the SGLS column and remove it from the other dfs
        sgls = list_df[0][["GDLS"]]
        for i in range(len(list_df)):
            list_df[i] = list_df[i].iloc[:, :-1]
        list_df.append(sgls)
    
    #If the experiment is Convergency 
    else:
        columns_names = CONVERGENCY[0:4]
        for conf in acronyms:
            for method in convergency_acronyms:
                new_columns.append(conf + "-" + method)
        new_columns+=CONVERGENCY[4:]

        #Extract the BFGS and SGLS columns and we remove it from the other dfs
        sgls = list_df[0][["GDLS"]]
        bfgs = list_df[0][["BFGS"]]
        for i in range(len(list_df)):
            list_df[i] = list_df[i].iloc[:, :-2]
        list_df.append(bfgs)
        list_df.append(sgls)

    #Create the final dataframe
    all_methods = pd.concat(list_df, axis=1)
    all_methods.columns = new_columns
    return all_methods, new_columns

def main():
    global PALETTE

    #ARGS
    args = parser.parse_args()
    path = Path(args.path)

    #Variables arguments
    time = args.time
    iters = args.iterations
    iters_per_sec = args.iters_per_seconds
    save_plot = args.saveplot
    convergence_flag = args.convergence_flag

    #Experiments arguments
    convergency = args.convergency
    r1 = args.robust1
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
        VAR_name = "Time"
    elif iters:
        VAR = "iterations"
        if r2:
            VAR = "Iterations"
        VAR_name = "Iterations"
    elif convergence_flag:
        VAR_name = "Convergence percentage"

        if r2: VAR = "Percentage of convergence"            

        #Name of the variable in Convergency and Robust I
        else: VAR = "Archived Convergence"

    elif iters_per_sec:
        VAR = "Iterations per Second"
        VAR_name = VAR

    #Not valid variable to analyze
    else:
        return -1

    #Experiments sets creation
    if convergency:
        EXP = "CONV"
        exp_name = "Convergence"
        final_table, problems = convergence(path, VAR)

    elif r1:
        #Assert the existence of the measured variable in this experiment
        assert VAR != "Iterations per Second", "The experiment 'Robust I' does not have the variable It/Sec"

        #Experiments variables
        EXP = "ROBUSTI"
        exp_name = "Robust I"

        #Variables usage name
        VAR = "Convergence_percentage"
        VAR_name = "Convergence Percentage"

        #Which methos were used (configuration)
        CONF = "All-Methods"
        conf_name = "all methods"

        #Dictionary with the values of the plot to construct 
        plot_variables = {"CONF": conf_name, "VAR": VAR_name, "EXP": exp_name}

        data = robust_I(path, variable=VAR) #This returns the DF for all configs, not a dictionary

        #Application of methods to the data
        colors_choiced = ["#38485C"] + (['#D55745', '#F6C443', '#5195D4']*3)
        PALETTE = dict(zip(data.columns, colors_choiced))
        plotting_heat_map(data, save_plot, False)
        plotting_friedmantest(data, VAR, EXP, CONF, PALETTE, saveplot=save_plot, less_is_better=False, plot_conf=plot_variables)
        return 0
    elif r2:
        #Assert the existence of the measured variable in this experiment
        assert VAR != "Iterations per Second", "The experiment 'Robust I' does not have the variable It/Sec"

        #Information of the experiment
        EXP = "ROBUSTII"
        exp_name = "Robust II"
        problems = existing_folders(path=path)
        final_table = robust_II(path, problems, VAR)
    else:
        return -1

    #Configuration to use (the final table contains a dict with key the conf and value their df)
    if flag_simplest:
        CONF = "simplest"
        conf_name = CONF + " configuration"
    elif flag_orignal:
        CONF = "original"
        conf_name = CONF + " configuration"
    elif flag_addedLS:
        CONF = "addedLS"
        conf_name = "with ls coonfiguration"
    elif flag_all:
        CONF = "All-Methods"
        conf_name = "all configurations"
    else:
        return -1             

    #Dictionary with the values of the plot to construct 
    plot_variables = {"CONF": conf_name, "VAR": VAR_name, "EXP": exp_name}

    #We have to manage the concatenation of the resulting DFs
    if CONF in ["simplest", "original", "addedLS"]:

        #Creation ranking and Friedman's Test
        data = final_table[CONF]
        data.index = problems

        if convergence_flag or iters_per_sec:
            plotting_friedmantest(data, VAR_name, EXP, CONF, PALETTE, save_plot, False, plot_variables)
        else:
            plotting_friedmantest(data, VAR_name, EXP, CONF, PALETTE, save_plot, True, plot_variables)

        if iters_per_sec:
            plotting_evolution(data, PALETTE, save_plot) 
        return 0

    #All the methods at once
    else:
        #Creation of plot of the Friedman test
        data = final_table 

        #Convergency
        if not r2:
            #Creation of the correct dataframe
            data, columns_name = process_all(data, False)
            data.index = problems

            #We put the dataframe in order. This order is defined to their dimension
            data['orden'] = data.index.map(PROBLEM_DIM_DICT)
            data = data.sort_values('orden').drop(columns=['orden'])

            #We define the colors for this scenario 
            colors_choiced = ['#D55745', '#F6C443', '#5195D4', '#58B89D']*3 + ["#4E247B", '#38485C']
            palette = dict(zip(columns_name, colors_choiced))

        #Robust Noise
        else: 
            #We process the data of all the configurations
            data, columns_name = process_all(data, True)

            #We define the colors for this case
            colors_choiced = ['#D55745', '#F6C443', '#5195D4']*3 + ["#38485C"]
            palette = dict(zip(columns_name, colors_choiced))
        
        #Modification of the plotting to adapt the flag where less is better
        if convergence_flag:
            plotting_heat_map(data, save_plot, False, "R2") 
            plotting_friedmantest(data, VAR_name, EXP, CONF, palette, saveplot=save_plot, less_is_better=False, plot_conf=plot_variables)
        elif iters_per_sec:
            plotting_evolution(data, palette, save_plot) 
            plotting_heat_map(data, save_plot, False, "C", True) 
            plotting_friedmantest(data, VAR_name, EXP, CONF, palette, saveplot=save_plot, less_is_better=False, plot_conf=plot_variables)
        else:
            plotting_friedmantest(data, VAR_name, EXP, CONF, palette, save_plot, True, plot_variables)


    #We print the information for the total number of instances of the variable
    print(data.head(59))

    #Cleaning of the data
    inf_mask = np.isinf(data).any(axis=1)
    data = data.replace([np.inf],0)
    cleaned_data = data[~inf_mask]
    total_per_method = cleaned_data.sum()

    #Chance the seconds to minutes
    total_per_method_hours = total_per_method / 60.0

    print(total_per_method_hours)
    return 0

if __name__ == "__main__":
    main()    
    