"""
Script intention.

This script has the intention to anmmend the error made it in the execution of the
convergency experiments. The BFGS had to be implemented used a line search that satisfies
the Strong Wolfe Condition (SWC). Nevertheless, we execute the experiments using Armijo Backtracking,
that does not guarantie the curvature condition in the BFGS derivation.

In this sense, the BFGS had a bad implementation. We already have solve it through the implementation
of a Line Search algorithm that satisfies the SWC. However, with the new implementation we need to execute all
the experiments all again. The experiments, took so much time in the past and we don't want to use the main.jl
function to only chance one results. Therefore, this script is solves that issue.

We allow the execution of one of the methods in some configuration. In other words, we do not have
the same options that in the main.jl to perform different kind of experiments. Then, the --modifierH, 
--modifierS, --useLS, are changed for two new parser options --config and --method. The execution 
of such method are stored in the following path:

csvs/results/METHOD/CONF/PROBLEM.csv

where METHOD, CONF, and PROBLEM are user-provide it parameters. Later other python script update the results using
the information in the METHOD path with respect the csvs/results/CONF (which are the results from the main.jl executions).

# Remark
Due to time limitations, and this script intention. We only implement the case for the BFGS method over all
configurations. Nevertheless, we left all the structure to complete the code for other methods. 

# Information contact:
Email: jose.quiroz@cimat.mx
Github-user: @Qntmth-uv | @Qntmth
Creation date:  June 2 26
"""
#Libs
using ArgParse

#Own Libs
include("NAMGM_methods.jl")
include("hessian_mod.jl")
include("utils.jl")


function parse_command_line()
    s = ArgParseSettings()
    s.description = "Auxiliar script to amend errors in execution of the main script (main.jl)"

    @add_arg_table! s begin
        "--problem"
            help = "SIF problem to solve"
            arg_type = String
            required = true

        "--show_info"
            help = "Show info of the problem a boolean flag (true if it's present)"
            action = :store_true

        "--varP"
            help = "Number of variables in the optimization problem (default -1,
            which means that the problem does not have other dimensions definitions)"
            arg_type = Int
            default = -1

        "--tol"
            help = "Minimum acceptable gradient norm"
            arg_type = Float64
            default = 1.e-8

        "--seed"
            help = "Fix a seed for the random process"
            arg_type = Int
            default = 0

        "--epsilon"
            help = "Epsilon added to the Modifier in case of being needed"
            arg_type = Float64
            default = 1.e-8
        
        "--config"
            help = "Configuration of Hessian modifier and LS to use (simplest, original, withLS)"
            arg_type = String
            required = true
        
        "--method"
            help = "Method to execute the problem (C-NAMGMS/SGLS/BFGS/Newton)"
            arg_type = String
            required = true
        
        "--nIters"
            help = "Maximum number of iterations of the method"
            arg_type = Int
            default = 1000
    end
   return parse_args(s) 
end


function main()
    #Load the program parameters
    args = parse_command_line()

    #Definite the code variables (problem characteristics)
    problem = args["problem"]
    nIters = args["nIters"]
    var_problem = args["varP"]
    tol = args["tol"]
    method = args["method"]
    config = args["config"]

    #Behavior variables (execution characteristics)
    seed = args["seed"]
    epsilonAdded = args["epsilon"]
    show_info = args["show_info"]
    
    #Path where to sava the results
    work_directory = pwd()
    path_results = joinpath("csvs/results", method, config)
    mkpath(joinpath(work_directory, path_results)) #Place where to save the results

    #Get the CUTEST functions (according to the dimension variable)
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem, var_problem)
    
    #Initialize random vector of same dimension
    n = length(initial_point)
    x0 = initial_point

    #Header of the data frame
    headers = ["iterations", "Last Gradient", "Execution time", "Iterations per Second", "Archived Convergence"]

    #Methods
    last_point, taken_iters, taken_time, last_norm, iters_per_seconds, convergence_flag = BFGSMethod(f, g, h, x0, tol, nIters)

    #Information to save
    data = [taken_iters, last_norm, taken_time, iters_per_seconds, convergence_flag]
    
    #Save the CSV file
    #df = DataFrame(data, headers)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end