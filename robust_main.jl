using ArgParse
using Random
using Distributions
using CUTEst

"""
# Intention of this script

Code to execute the second series of experiments proposed in my master's degree thesis.

The purpose of this experiments is to investigate how the proposed methods work under 
far initial points. Normally all the given points in the optimization problems are 
relatively close to the solution. This is not the usage case of most of the optimization
algorithms; where usually starts on a random position (for instance DNNs). 

# What it does

The methodology used to investigate that is the famous paper 'testing optimization software (1981)' 
by Jorge J. More et. al. They proposes a simple testing schema. It consist of using the given initial
point X0 and moving it by a factor of 10 on each new execution of the algorithms (this is NTRIES variable).
There is not a standard value for NTRIES, it depends completely on the optimization problem.

This script execute the before mentioned methodology, and saves the results in each new execution. 
For that we create a directory of results for the optimization problem (not optional), the path were all the results
of the executions is

pwd/csv/results/robust/PROBLEM_NAME/* #(csvs/results/robust/ is hardcoded)

and the result are named as

PROBLEM_FACTOR_X.csv

Where pwd is the current execution folder, PROBLEM_NAME is the name of SIF file (without the .sif extension)
and X is the factor used in the experiments (determined by the NTRIES hyperparameter).

# Side notes

All the used set of configurations and problems that used this script are located in the path 'exe_commands/robust/*'

There are parameters for the execution of the optimization problems, please read the parser parameters.

This clearly can be implemented in the 'main.jl' this methodology, but right now that code is kinda messy, so I created 
another new script to not mess more that script.

# Contact information

Contact: jose.quiroz@cimat.max
Date: April 2026

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
            default = "none"

        "--modifierS"
            help = "System modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag','remove'}"
            arg_type = String
            default = "none"
        
        "--lqueue"
            help = "Maximum number of elements in the Queue Gradients - NAMGMG"
            arg_type = Int
            default = 3
        
        "--NTRIES"
            help = "Number of times that the init point will be multiplied by 10"
            arg_type = Int
            default = 1

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
        
        "--subdirectory"
            help = "Subdirectory where to save the different results"
            arg_type = String
            default = "original/"
        end
    return parse_args(s)
end

function main()

    #Load the arguments
    parsed_args = parse_commandline()
    
    #Principal information of the script
    problem = parsed_args["problem"]
    show_info = parsed_args["show_info"]
    NTRIES = parsed_args["NTRIES"]
    name = first(splitext(basename(problem)))
    varN = parsed_args["varN"] 
    varP = parsed_args["varP"]
    subdirectory = parsed_args["subdirectory"]

    #Parameters of the method
    nIters = parsed_args["nIters"]
    lqueue = parsed_args["lqueue"]
    randomsize = lqueue
    tol = parsed_args["tol"]
    seed = parsed_args["seed"]
    epsilonAdded = parsed_args["epsilon"]
    use_LS = parsed_args["useLS"]
    repetitions = parsed_args["repetitions"]
    
    #Covert the modifier name in lowercase (the function get_modifier matches lowercases strings)
    modH = lowercase(parsed_args["modifierH"]) 

    #Path were the results will be saved. We create a folder for each problem inside the subdirectory
    #In this case all the results will be saved in ./csvs/results/subdirectory/PROBLEM_NAME/*
    factors_folder_path = joinpath("csvs/results/robust", subdirectory, name)
    
    #We create the paths in case that not exists
    working_directory = pwd()
    mkpath(joinpath(working_directory, factors_folder_path))

    #File where we save the results
    file_basis = joinpath(factors_folder_path, name*"_FACTOR_")

    #Print the values in the parser (problem configuration)
    println("-"^40)
    @printf("%-20s | %-20s\n", "Parameters", "Value")
    println("-"^40)

    #Convert the Dict in a set of bidimensional vectors sorted by the keys
    for (key) in sort(collect(keys(parsed_args)))
        @printf("%-20s | %-20s\n", key, parsed_args[key])
    end
    println("-"^40)
    
    #Fix a Seed for generation
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Obtain the asked Hessian modifier
    modH = get_modifier(modH)
    headers = [ "Archived Convergence", "Iterations", "Execution time", "Last Gradient", "Not divergency number"]

    #Get the problem functions and initialize the variables for the robust optimization
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem, varP, varN)
    x0 = initial_point
    Factor = 1.0


    #Number of executions that will be executed
    for i in (1:NTRIES)

        #Variables where store the results of the current iteration.
        U = Any[0, 0.0, 0.0, 0.0, 0.0]
        M = vcat(fill(U', repetitions)...)
        divergent_results::Int = 0;
        solve_most_of_problems = nothing

        #Show information of the evolution of the factor if it is required
        show_info ? println("Factor: $Factor") : nothing

        #Execution of the NAMGM methods
        xf_Oviedo, iterOviedo, t_oviedo, normOviedo, ttpSO, flagOviedo = namgmOviedo(f, g, h, x0, tol, nIters, modH, epsilonAdded, use_LS)    
        xf_Queue, iterQueue, t_queue, normQueue, ttpSQ, flagQueue = namgmGrads(f, g, h, x0, tol, nIters, lqueue, modH, epsilonAdded, use_LS)
        for i in 1:repetitions
            M[i, :] .= namgmRandomVectors(f, g, h, x0, tol, nIters, lqueue-1, modH, epsilonAdded, show_info, use_LS)[2:end]
            if isinf(M[i, 3]) || M[i, 3] > 1.0^12
                divergent_results+=1;
            else
                U.+=M[i, :]
            end
        end

        
        #Compute the mean of the results for the Random - NAMGM.
        total_successful_experiments = repetitions-divergent_results
        U./=(total_successful_experiments)
        iterRandom, t_random, normRandom, ttpSR, flagRandom = U
        solve_most_of_problems = (flagRandom >= 0.5) ? true : false
        displayResults("Random Mean ($total_successful_experiments repetitions)", iterRandom, t_random, normRandom, ttpSR, solve_most_of_problems)
        println("The number of divergent results in Random - NAMGM: $divergent_results")

        #Creation of the columns for the results
        times = [t_oviedo, t_queue, t_random]
        lastGrad = [normOviedo, normQueue, normRandom]
        iterations =[iterOviedo, iterQueue, iterRandom]
        convergence = [flagOviedo, flagQueue, flagRandom] 
        times_that_diverged = [isinf(normOviedo) ? 0 : 1, isinf(normQueue) ? 0 : 1, Float64(total_successful_experiments)] 
        data = [convergence, iterations, times, lastGrad, times_that_diverged]
        
        #Save the information of the current file
        current_file = file_basis*string(i)*".csv"

        #Save the CSV file
        df = DataFrame(data, headers)
        CSV.write(current_file, df)

        #Information of the saved files
        println("Saved results at: $current_file")

        #Factor the variables for the next execution
        x0 *= 10.0
        Factor *= 10.0
    end



    #Finalize the model
    finalize(nlp_problem)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end