using ArgParse
using Random
using Distributions
using CUTEst

"""
Execution examples

#https://www.sfu.ca/~ssurjano/beale.html
julia robust_main.jl --problem cutest-sif/BEALE.SIF --LB -4.5 --UB 4.5 

# https://www.sfu.ca/~ssurjano/hart6.html
julia robust_main.jl --problem cutest-sif/HEART6LS.SIF --LB 0.0 --UB 1

# https://www.sfu.ca/~ssurjano/rosen.html
julia robust_main.jl --problem cutest-sif/ROSENBR.SIF --LB -5.0 --UB 10.0
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

    #Path were the results will be saved
    path_results = joinpath("csvs/results/robust", subdirectory)
    workdirectory = pwd()
    mkpath(joinpath(workdirectory, path_results)) #Place where to save the results 

    #Print the values of the parser
    println("-"^40)
    @printf("%-20s | %-20s\n", "Parameters", "Value")
    println("-"^40)

    #Convert the Dict in a set of bidimensional vectors sorted by the keys
    for (key) in sort(collect(keys(parsed_args)))
        @printf("%-20s | %-20s\n", key, parsed_args[key])
    end
    println("-"^40)

    #methods
    modH = lowercase(parsed_args["modifierH"])
    modS = lowercase(parsed_args["modifierS"])
    repetitions = parsed_args["repetitions"]
    
    #File where we save the results
    file_basis = joinpath(path_results, name*"_FACTOR_")

    #Generator of the uniform distribution
    # LB = parsed_args["LB"]
    # UB = parsed_args["UB"] 

    #Draw from a uniform given the box where are evaluating
    #d = Uniform(LB, UB)

    #Fix a Seed for generation
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Obtain the used modifier
    modH = get_modifier(modH)
    headers = [ "Archived Convergence", "Iterations", "Execution time", "Last Gradient"]

    #Get the CUTEST functions    
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem, varP, varN)
    x0 = initial_point
    Factor = 1.0
    #println("Initial point: ", initial_point)

    #Number of executions that will be executed
    for i in (1:NTRIES)

        #Variables where store the results of the current iteration.
        U = Any[0, 0.0, 0.0, 0.0, 0.0]
        M = vcat(fill(U', repetitions)...)
        solve_most_of_problems = nothing

        println("Factor: $Factor")
        
        #Execution of the NAMGM methods
        xf_Oviedo, iterOviedo, t_oviedo, normOviedo, ttpSO, flagOviedo = namgmOviedo(f, g, h, x0, tol, nIters, modH, epsilonAdded, use_LS)
        xf_Queue, iterQueue, t_queue, normQueue, ttpSQ, flagQueue = namgmGrads(f, g, h, x0, tol, nIters, lqueue, modH, epsilonAdded, use_LS)
        for i in 1:repetitions
            M[i, :] .= namgmRandomVectors(f, g, h, x0, tol, nIters, lqueue-1, modH, epsilonAdded, show_info, use_LS)[2:end]
            U.+=M[i, :]
        end
        
        #Compute the mean of the results for the Random - NAMGM Method.
        U./=repetitions
        iterRandom, t_random, normRandom, ttpSR, flagRandom = U
        solve_most_of_problems = (flagRandom >= 0.5) ? true : false
        displayResults("Random Mean ($repetitions repetitions)", iterRandom, t_random, normRandom, ttpSR, solve_most_of_problems)

        #Creation of the columns for the results
        times = [t_oviedo, t_queue, t_random]
        lastGrad = [normOviedo, normQueue, normRandom]
        iterations =[iterOviedo, iterQueue, iterRandom]
        convergence = [flagOviedo, flagQueue, flagRandom] 
        data = [convergence, iterations, times, lastGrad]

        #Save the information of the current file
        current_file = file_basis*string(i)*".csv"

        #Save the CSV file
        df = DataFrame(data, headers)
        CSV.write(current_file, df)

        #Information of the saved files
        println("Saved results at: $current_file")

        # Draw a random vector from a standard normal
        x0 *= 10.0
        Factor *= 10.0


    end

    #Finalize the model
    finalize(nlp_problem)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end