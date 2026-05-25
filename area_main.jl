using ArgParse
using Random
using Distributions
using CUTEst

"""
# Intention of this script

Code to execute a third series of experiments proposed in my master's degree thesis.

The purpose of this experiments is to investigate how the proposed methods work under a small modification
of the initial point given by a normal. This scenario is more realistic that the researched in the second 
set of experiments, due that the initial points often have small round off errors.

# What it does

This script calls a method and then execute 'exp_rep' repetitions of the experiment. In each iteration, the initial
point is modified by a vector P such that P∼N(0, σI), where σ=√dim(x0). It's allowed to not create a folder to save
the results, nevertheless is encourage. If is asked to save the results of this experiments, then those are saved in
the following two paths

(1) pwd/csv/results/area_robust/PROBLEM/subdirectory #(csvs/results/area_robust/ is hardcoded)
(2) pwd/csv/historial/area_robust/PROBLEM/subdirectory  #(csvs/historial/area_robust/ is also hardcoded)

and the results are named as

(1) METHOD.csv
(2) METHOD.csv 

Where pwd is the current execution folder, PROBLEM_NAME is the name of SIF file (without the .sif extension)
and METHOD is the name of the used method (determined by the --method flag; see the parser args for more information).

# Side notes

All the used set of configurations and problems that used this script are located in the path 'exe_commands/area_robust/*'

There are parameters for the execution of the optimization problems, please read the parser parameters.

This file was write after the 'robust_main.jl', that is why there are two files to make similar things.

This file needs other variable '--method', this is aimed to run in parallel diverse problems. In other scripts as
'main.jl' and 'robust.jl' runs all the methods in one execution. This make the code less executable in parallel 
for different configurations and methods. Maybe to accelerate the results, this script will be sended into the INSURGENTE.

There are intensions to create a file to observe how to different points try to converge (just for bidimensional functions)

# Contact information

Contact: jose.quiroz@cimat.mx
Alias: @Qntmth || @Qntmth-uv (Github)
Date: May 1, 2026
"""

include("utils.jl")
include("NAMGM_methods.jl")

function parse_commandline()
    s = ArgParseSettings()
    s.description = "Main Script of the set robust experiments of the NAMGM method."
     
    @add_arg_table! s begin
        "--problem"
            help = "SIF problem to solve"
            arg_type = String
            required = true

        "--subdirectory"
            help = "Subdirectory where to save the different results"
            arg_type = String
            required = true

        "--show_info"
            help = "Show info of the problem a boolean flag (true if it's present)"
            action = :store_true

        "--repetitions"
            help = "Number of repetitions for the Random method. " 
            arg_type = Int64
            default = 30

        "--epsilon"
            help = "Epsilon added to the Modifier in case of being needed"
            arg_type = Float64
            default = 1.0e-8

        "--seed"
            help = "Fix a seed for the random process"
            arg_type = Int
            default = 0

        "--nIters"
            help = "Maximum number of iterations of the method"
            arg_type = Int
            default = 1000

        "--tol"
            help = "Minimum acceptable gradient norm"
            arg_type = Float64
            default = 1.0e-8

        "--modifierH"
            help = "Hessian modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag', 'remove'}"
            arg_type = String
            default = "eigen"

        "--modifierS"
            help = "System modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag','remove'}"
            arg_type = String
            default = "none"
        
        "--lqueue"
            help = "Maximum number of elements in the Queue Gradients - NAMGMG"
            arg_type = Int
            default = 3
        
        "--useLS"
            help = "Use lines search in the tested methods (backtracking)"
            action = :store_true
        
        "--varP"
            help = "Number of variables in the optimization problem (default -1,
            which means that the problem does not have other dimensions definitions)"
            arg_type = Int
            default = -1

        "--varN"
            help = "Name of the dimension variable (it is not a standard name, most of the problems assigns the
            dimension variable 'N', but others assign the variable 'n'; e.g., WOODS)"
            arg_type = String
            default = "N"

        "--method"
            help = "Method to be used in the execution of the test. Available: {SGLS, AMG, Queue, Random} (default='AMG')"
            arg_type = String
            default = "AMG"

        "--exp_rep"
            help = "Number of repetitions of the experiment, the number elements to be drawn from the distributions of X0"
            arg_type = Int
            default = 30

        "--saveinfo"
            help = "Boolean variable to indicate if it should write files"
            action = :store_true 
        end
    return parse_args(s)
end


function main()

    #Load the arguments
    parsed_args = parse_commandline()
    
    #Principal information of the script
    problem = parsed_args["problem"]
    show_info::Bool = parsed_args["show_info"]
    name = first(splitext(basename(problem)))
    varN = parsed_args["varN"] 
    varP = parsed_args["varP"]
    subdirectory = parsed_args["subdirectory"]
    save_info::Bool = parsed_args["saveinfo"]

    #Parameters of the method
    nIters = parsed_args["nIters"]
    lqueue = parsed_args["lqueue"]
    tol = parsed_args["tol"]
    seed::Int = parsed_args["seed"]
    epsilonAdded = parsed_args["epsilon"]
    use_LS::Bool = parsed_args["useLS"]
    elements_to_draw::Int = parsed_args["exp_rep"]
    repetitions::Int = parsed_args["repetitions"]
    
    #Covert the modifier name in lowercase (the function get_modifier matches lowercases strings)
    modH = lowercase(parsed_args["modifierH"]) 
    method = lowercase(parsed_args["method"]) 
    
    #Assertion of correct method usage
    if method ∉ ["sgls", "amg", "queue", "random"]
        println("Not valid method. The available methods are ['SGLS', 'AMG', 'Queue', 'Random']")
        return -1
    end

    #Path were the results will be saved. We create a folder for each problem inside the subdirectory
    #In this case all the results will be saved in ./csvs/results/PROBLEM_NAME/subdirectory/*
    factors_folder_path = joinpath("csvs/results/area_robust", name, subdirectory)
    historial_folder_path = joinpath("csvs/historials/area_robust", name, subdirectory) 
    
    #We create the paths in case that not exists
    working_directory = pwd()

    #We create the paths in case of saving the results
    if save_info
        mkpath(joinpath(working_directory, factors_folder_path))
        mkpath(joinpath(working_directory, historial_folder_path))
    end


    #File where we save the results (The name file of the results)
    file_basis = joinpath(factors_folder_path, method)
    historial_file = joinpath(historial_folder_path, method)

    #Print the values in the parser (problem configuration)
    println("-"^40)
    @printf("%-20s | %-20s\n", "Parameters", "Value")
    println("-"^40)

    #Convert the Dict in a set of bidimensional vectors sorted by the keys
    for (key) in sort(collect(keys(parsed_args)))
        @printf("%-20s | %-20s\n", key, parsed_args[key])
    end
    println("-"^40)
    
    #Fix a seed to make repeatable the experiments
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Obtain the asked Hessian modifier
    modH = get_modifier(modH)

    #Get the problem functions and initialize the variables for the robust optimization
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem, varP, varN)
    x0 = initial_point
    dimension = length(x0)
    std = sqrt(dimension)

    #Variables where store the results of the current iteration.
    U = Any[0, 0.0, 0.0, 0.0, 0.0] #Place to save the means
    M = vcat(fill(U', elements_to_draw)...) #Matrix to save each iteration results
    divergent_results::Int = 0;
    solve_most_of_problems = nothing

    #Number of times that we will drawn a point 
    for i in (1:elements_to_draw)

        #We drawn a vector P_i to simulate noise in the input X0
        vector_noise = rand(Normal(0.0, std), dimension)
        x0.+=vector_noise

        #Method to execute
        """
        The election of the methods its not efficient, but we do not care so much.
        (This could be solved using function name(; arguments, kwards...) but right now 
        it may require the re-factorization of the code. Given that is used in main.jl and robust_main.jl)
        Maybe if there is time, I can re-factorize it.    
        -    @Qntmth - May 1, 2026.
        """
        print("| $i |")
        if method == "sgls"
            M[i, :] .= steepestMethod(f, g, x0, tol, nIters)[2:end]
        elseif method== "amg"
            M[i, :] .= namgmOviedo(f, g, h, x0, tol, nIters, modH, epsilonAdded, use_LS)[2:end]
        elseif method == "queue"
            M[i, :] .= namgmGrads(f, g, h, x0, tol, nIters, lqueue, modH, epsilonAdded, use_LS)[2:end]
        elseif method == "random"
            #Variables where store the results of the current iteration for the Random-Namgm.
            V = Any[0, 0.0, 0.0, 0.0, 0.0]
            N = vcat(fill(U', repetitions)...)
            divergent_results_random::Int = 0 

            #Inner repetitions of the experiments, given that Random - NAMGM is stochastic.
            for j in 1:repetitions
                N[j, :] .= namgmRandomVectors(f, g, h, x0, tol, nIters, lqueue-1, modH, epsilonAdded, use_LS, show_info)[2:end]

                #Assertion that the results are not divergent
                (isinf(N[j, 3]) || N[j, 3] > 1.0e12) ? divergent_results_random+=1 : V.+=N[j, :]
            end

            #Get the mean of the results for the Random method
            total_successful_experiments_random::Int = 0;
            if divergent_results_random == elements_to_draw
                V = [0.0, Inf, Inf, Inf, 0.0]
            else
                total_successful_experiments_random = repetitions-divergent_results_random
                V./=(total_successful_experiments_random)
                V[5] *= total_successful_experiments_random/repetitions
            end
            
            solve_most_of_problems_random = (V[5] >= 0.5) ? true : false
            displayResults("Random Mean ($total_successful_experiments_random repetitions)", V[1], V[2], V[3], V[4], solve_most_of_problems_random) 
            #println("The number of divergent results Random - NAMGM: $divergent_results_random")

            #Add the mean of the results
            M[i, :] .= V
        end

        #Verification that the results are clean (in other words, that the method does not diverge)
        if isinf(M[i, 3]) || M[i, 3] > 1.0e12
            divergent_results+=1;
        #If the method does not diverged, then we add the results
        else
            U.+=M[i, :]
        end 
    end 
    total_successful_experiments::Int = 0;
    #Compute the mean of the results for the method and saved it.
    if divergent_results == elements_to_draw
        U = [0.0, Inf, Inf, Inf, 0]
    else
        total_successful_experiments = elements_to_draw-divergent_results
        U./=(total_successful_experiments)
        U[5] *= total_successful_experiments/elements_to_draw
    end

    #Display information of the experiment
    iters, time, norm, itersPerSec, convergence_percent = U
    solve_most_of_problems = (convergence_percent >= 0.5) ? true : false
    displayResults("$method Mean ($total_successful_experiments repetitions)", iters, time, norm, itersPerSec, solve_most_of_problems)
    println("The number of divergent results in method $method: $divergent_results")

    #Save the information
    headers = [ "Percentage of convergence", "Iterations", "Execution time", "Last Gradient", "Not divergency number"]
    headers_historial = ["Iterations", "Execution time", "Last Gradient", "Itters Per Second* ", "Percentage of convergence"] 
    data = [convergence_percent iters time norm total_successful_experiments]
    
    #Save the information of the current file
    current_file = file_basis*".csv"
    history_file = historial_file*".csv"

    #Save the CSV file
    df = DataFrame(data, headers)
    dfh = DataFrame(M, headers_historial)

    if save_info
        CSV.write(current_file, df)
        CSV.write(history_file, dfh)
    end

    #Information of the saved files
    println("Saved results at: $current_file")
    println("The results of the execution were saved at: $historial_file")
    
    #Finalize the model
    finalize(nlp_problem)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end