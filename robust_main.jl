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
            default = 1.e-8

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
            default = 1.e-8

        "--modifierH"
            help = "Hessian modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag'
                    'remove'}"
            arg_type = String
            default = "none"

        "--modifierS"
            help = "System modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag'
                    'remove'}"
            arg_type = String
            default = "none"
        
        "--lqueue"
            help = "Maximum number of elements in the NAMGMGradQueue"
            arg_type = Int
            default = 3

        "--LB"
            help= "Lower bound where to draw from a Uniform distribution"
            arg_type = Float64
            default = -1.0

        "--UB"
            help= "Upper bound where to draw from a Uniform distribution"
            arg_type = Float64
            default = 1.0
    end
    return parse_args(s)
end

function main()

    #Load the arguments
    parsed_args = parse_commandline()
    
    #Principal information
    problem = parsed_args["problem"]
    show_info = parsed_args["show_info"]
    
    #Parameters of the method
    nIters = parsed_args["nIters"]
    lqueue = parsed_args["lqueue"]
    randomsize = lqueue
    tol = parsed_args["tol"]
    seed = parsed_args["seed"]
    epsilonAdded = parsed_args["epsilon"]

    modH = lowercase(parsed_args["modifierH"])
    modS = lowercase(parsed_args["modifierS"])
    repetitions = parsed_args["repetitions"]
    LB = parsed_args["LB"]
    UB = parsed_args["UB"] 

    #Fix a Seed for generation
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Obtain the used modifier
    modH = get_modifier(modH)

    #Get the CUTEST functions    
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem)
    N = length(initial_point)

    #Print information
    println("Initial point: ", initial_point)
    println("Interval to drawn from a Uniform Distribution: [$LB, $UB]")
    
    #Draw from a uniform given the box where are evaluating
    d = Uniform(LB, UB)

    #Variables where store the results of the current iteration.
    U = Any[0, 0.0, 0.0, 0.0, 0.0]
    M = vcat(fill(U', repetitions)...)
    solve_most_of_problems = nothing

    for i in (1:repetitions)
        # Draw a random vector from a standard normal
        x0 = rand(d, N)

        #New point where to evaluate
        println("Drawn Point: ", x0)

        #Execute the algorithms
        M[i, :] .= namgmOviedo(f, g, h, x0, tol, nIters, modH, epsilonAdded, false)[2:end]
        U.+=M[i, :]
    end

    #Take the mean of the results
    U./=repetitions

    #Assign the variables
    iterOviedo, t_oviedo, normOviedo, ttpSO, flagOviedo = U
    solve_most_of_problems = (flagOviedo >= 0.5) ? true : false

    #Show the results of the method
    displayResults("Oviedo Mean ($repetitions repetitions)",
                    iterOviedo, t_oviedo, normOviedo, ttpSO, solve_most_of_problems)

    #Finalize the model
    finalize(nlp_problem)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end