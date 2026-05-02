using Random
using ArgParse

"""
Execution command (example)
julia --project=venv_NAMGM main.jl --problem cutest-sif/ROSENBR.SIF --nIters 5 --modifier eigen 
julia --project=venv_NAMGM main.jl --problem cutest-sif/ROSENBR.SIF --modifier eigen 
julia --project=venv_NAMGM main.jl --problem cutest-sif/DECONVU.SIF --seed 2 --useDimProblem --DEBUG  
"""

const minValue = 2.220446049250313e-16


include("NAMGM_methods.jl")
include("hessian_mod.jl")
include("utils.jl")
include("plotting_util.jl")
include("DEBUG_methods.jl")

function parse_commandline()
    s = ArgParseSettings()
    s.description = "Main Script of the set of experiments of the NAMGM method."

     
    @add_arg_table! s begin
        "--problem"
            help = "SIF problem to solve"
            arg_type = String
            required = true
        "--show_info"
            help = "Show info of the problem a boolean flag (true if it's present)"
            action = :store_true
        "--image_name"
            help = "Name of the result plot"
            arg_type = String
            default = "default"
        "--file_name"
            help = "Name of the output file (CSV)"
            arg_type = String
            default = "default"
        "--nIters"
            help = "Maximum number of iterations of the method"
            arg_type = Int
            default = 1000
        "--varP"
            help = "Number of variables in the optimization problem (default -1,
            which means that the problem does not have other dimensions definitions)"
            arg_type = Int
            default = -1
        "--lqueue"
            help = "Maximum number of elements in the NAMGMGradQueue"
            arg_type = Int
            default = 3
        "--tol"
            help = "Minimum acceptable gradient norm"
            arg_type = Float64
            default = 1.e-8
        "--modifierH"
            help = "Hessian modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag'
                    'remove'}"
            arg_type = String
            default = "eigen"

        "--modifierS"
            help = "System modifier Strategy. Available: {'none', 'eigen', 'diag', 'sabsdiag', 'maxdiag', 'tridiag'
                    'remove'}"
            arg_type = String
            default = "none"

        "--seed"
            help = "Fix a seed for the random process"
            arg_type = Int
            default = 0

        "--epsilon"
            help = "Epsilon added to the Modifier in case of being needed"
            arg_type = Float64
            default = 1.e-8

        "--DEBUG"
            help= "Use the DEBUG functions of the methods to create CSV's and some informative Plots"
            action = :store_true
        
        "--useDimProblem"
            help= "Use the same number of vectors as the dimension of the problem (valid only on RandomVectors)"
            action = :store_true

        "--repetitions"
            help = "Number of repetitions for the Random method. " 
            arg_type = Int64
            default = 30

        "--saveinfo"
            help = "Boolean variable to indicate if it should write files"
            action = :store_true 

        "--useLS"
            help = "Use lines search in the tested methods (backtracking)"
            action = :store_true

        "--displaysG"
            help = "Show the possible generated graphs"
            action = :store_true
        
        "--subdirectory"
            help = "Subdirectory where to save the different results using different parameters (default, addedLS & usingDBFS)"
            arg_type = String
            default = "original/"

        "--dontuse_ModifierNewton"
            help = "It removes the given modifier of the Hessian and uses the none mode."
            action = :store_true

        "--factorX0"
            help = "Multiple the initial point by a factor (default = 1.0)"
            arg_type = Float64
            default = 1.0
            
        "--DM"
            help = "Prints the result points of the optimization process (ONLY IN DEBUGMODE)"
            action = :store_true


    end
    return parse_args(s)
end


function main()

    #Load the arguments
    parsed_args = parse_commandline()
    
    #Defined variables
    problem = parsed_args["problem"]
    show_info = parsed_args["show_info"]
    debug_mode = parsed_args["DEBUG"]
    usedim_problem = parsed_args["useDimProblem"] 
    image_name = parsed_args["image_name"]
    file_name = parsed_args["file_name"]
    nIters = parsed_args["nIters"]
    varproblem = parsed_args["varP"]
    lqueue = parsed_args["lqueue"]
    randomsize = lqueue
    tol = parsed_args["tol"]
    seed = parsed_args["seed"]
    epsilonAdded = parsed_args["epsilon"]
    modH = lowercase(parsed_args["modifierH"])
    modS = lowercase(parsed_args["modifierS"])
    repetitions = parsed_args["repetitions"]
    saveinfo = parsed_args["saveinfo"]
    use_LS = parsed_args["useLS"]
    show_plots = parsed_args["displaysG"]
    subdirectory = parsed_args["subdirectory"]
    dont_use_modifier_newton = parsed_args["dontuse_ModifierNewton"] 
    factor = parsed_args["factorX0"]  
    display_results = parsed_args["DM"]

    #The debug mode makes save info true
    saveinfo = debug_mode ? true : saveinfo

    #Fix a Seed for generation
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Name of the results
    name = first(splitext(basename(problem)))

    #If it is using LS, then change the save directory
    use_LS && subdirectory == "original/" ? subdirectory = "addedLS" : nothing

    #Images paths formatting
    if image_name == "default"
        images_path = "images/"
        
        #Paths of folders. In such directory will be created folders
        path_image_historial = joinpath(images_path, "historials_grads", subdirectory) 
        path_image_path = joinpath(images_path, "paths", subdirectory) 
        path_image_conditionEvo = joinpath(images_path, "condition_analysis", subdirectory)  

        #Images names to write
        image_historial = joinpath(path_image_historial, name*"_"*modH*".svg")
        image_path = joinpath(path_image_path, name*"_"*modH*"_path.svg") 
        image_conditionEvolution = joinpath(path_image_conditionEvo, name*"_"*modH*"CN_Evolution.png") 
    end

    #Files path formatting
    if file_name == "default"
        #Paths where to save the results
        path_results = joinpath("csvs/results", subdirectory)
        path_historials = joinpath("csvs/historials", subdirectory) 
        path_results_randoms = joinpath("csvs/results/random_historials", subdirectory)

        #Files to be write
        file_name_results = joinpath(path_results, name*"_"*modH*".csv")
        file_name_historials = joinpath(path_historials, name*"_"*modH*".csv")
        file_name_results_historials = joinpath(path_results_randoms, name*"_"*modH*".csv")
    else
        file_name_results = file_name
    end

    #Creation of the results directory, if there is not such path
    if saveinfo
        #Get the current working directory
        workdirectory = pwd()

        #Create directory in case it does not exists (Images)
        mkpath(joinpath(workdirectory, path_image_historial))
        mkpath(joinpath(workdirectory, path_image_path))
        mkpath(joinpath(workdirectory, path_image_conditionEvo))

        #Create directory in case it does not exist (FILES)
        mkpath(joinpath(workdirectory, path_results)) #Place where to save the results
        mkpath(joinpath(workdirectory, path_historials)) #Place where to save the historials of the Gradients
        mkpath(joinpath(workdirectory, path_results_randoms)) #Place where to save the historials of general results of the randoms executions
    end

    #Print the values of the parser
    println("-"^40)
    @printf("%-20s | %-20s\n", "Parameters", "Value")
    println("-"^40)

    #Convert the Dict in a set of bidimensioal vectors sorted by the keys
    for (key) in sort(collect(keys(parsed_args)))
        @printf("%-20s | %-20s\n", key, parsed_args[key])
    end
    println("-"^40)

    #println("Modifier Hessian: ", modH, " | Modifier System: ", modS)
    modH = get_modifier(modH)
    modS = get_modifier(modS)

    #We make that the approximation of the Newton Method is null, i. e., the approximation is the hessian matrix
    if dont_use_modifier_newton
        println("The Newton Method is set up as the with the Null/Trivial approximation of the Hessian, i. e., the Hessian Matrix ")
        modN = get_modifier("none")
    else
        modN = modH     
    end

    #Get the CUTEST functions (according to the dimension variable)
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem, varproblem)
    
    #Initialize random vector of same dimension
    n = length(initial_point)
    x0 = initial_point * factor

    #Check the flag of the size of RandomVectors
    usedim_problem ? randomsize = n : nothing
    println("Random Vectors being used: ", randomsize-1)
    println("The factor for the initial point is: $factor")
    println("-"^80)


    #Header of the DF
    headers = ["iterations", "Last Gradient", "Execution time", "Iterations per Second", "Archived Convergence"]
    headers_methods = ["AMG", "Gradient Queue", "Random", "Newton", "BFGS", "GDLS"]

    #Variables in a TRY-CATCH
    xf_newton, historial_newton, t_newton, xP_newton, iterNewton, normNewton, ttpSN, flagNewton = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing

    if debug_mode
        println(">>DEBUG Mode")

        #Run the NAMGM algorithms using DEBUG measurements
        xf_Oviedo, historial_Ovideo, t_oviedo, xP_oviedo, iterOviedo, normOviedo, CN_O, ttpSO, flagOviedo = DEBUG_namgmOviedo(f, g, h, x0, tol, nIters, modH, epsilonAdded, modS, use_LS)
        xf_Queue, historial_Queue, t_queue, xP_queue, iterQueue, normQueue, CN_Q, ttpSQ, flagQueue = DEBUG_namgmGrads(f, g, h, x0, tol, nIters, lqueue, modH, epsilonAdded, modS, use_LS)
        xf_random, historial_random, t_random, xP_random, iterRandom, normRandom, CN_R, ttpSR, flagRandom = DEBUG_namgmRandomVectors(f, g, h, x0, tol, nIters, randomsize, modH, epsilonAdded, modS, use_LS)
        M = [iterRandom, t_random, normRandom, ttpSR]
        try
            xf_newton, historial_newton, t_newton, xP_newton, iterNewton, normNewton, ttpSN, flagNewton= DEBUG_newtonMethod(f, g, h, x0, tol, nIters, modN, epsilonAdded, use_LS)
        catch e
            println("An error was found in the execution of the Newton method: $e")
            xf_newton, historial_newton, t_newton, xP_newton, iterNewton, normNewton, ttpSN, flagNewton= x0, [], Inf, [], Inf, Inf, 0.0, false
        end
        xf_bfgs, historial_bfgs, t_bfgs, xP_bfgs, iterBFGS, normBFGS, ttpSB, flagBFGS = DEBUG_BFGSMethod(f, g, h, x0, tol, nIters, show_info)
        xf_SGD, historial_sgd, t_sgd, xP_sgd, iterSGD, normSGD, ttpSGD, flagSGD = DEGUB_steepestMethod(f, g, x0, tol, nIters)

        #Displays the obtained solutions
        if(display_results)
            println("-"^80)
            println("Solutions of the optimization algorithms")
            println("."^80)
            display("SGLS Solution: $xf_SGD")
            display("AMG Solution: $xf_Oviedo")
            display("Queue Solution: $xf_Queue")
            display("Random Solution: $xf_random")
            display("BFGS Solution: $xf_bfgs")
            display("Newton Solution: $xf_newton")
        end

        #Clean the data if there is a 0.0 then the plot will explode.
        H = [historial_Ovideo, historial_Queue, historial_random, historial_newton, historial_bfgs, historial_sgd]
        Hc = [CN_O, CN_Q, CN_R]

        for i in 1:length(H)
            if !isempty(H[i])
                if isnan(H[i][end]) || H[i][end] == 0
                    H[i][end] = 0.0
                elseif (H[i] == Inf)
                    H[i][end] = 1e104
                end
            end
        end

        #It should save the data?
        saveinfo ? createCSV(H, file_name_historials, headers_methods) : nothing

        #Variables to plot the results
        problems_labels = ["AMG", "Queue", "RD", "Newton", "BFGS", "SGD"]
        colors_list = [:blue, :red, :purple, :green, :orange, :salmon]
        styles_list = [:solid, :dash, :dash, :solid, :dash, :solid]

        #Plot the results (The gradient historial and the condition number)
        plot_title = "Convergence Analysis: " * name
        plot_titleCN = "Condition Analysis: " * name
        gradplot = plotEvolution(H, colors_list, problems_labels, styles_list, "||∇f(x)||", plot_title)
        conditionplt = plotEvolution(Hc, colors_list[1:end-2], problems_labels[1:end-2], styles_list[1:end-2], "κ(Hψ)", plot_titleCN)
        
        #Save the images if it's required
        saveinfo ? savefig(gradplot, image_historial) : nothing 
        saveinfo ? savefig(conditionplt, image_conditionEvolution) : nothing 

        #If the function is bidimensional and we are saving the information, then we construct the plot of the followed path
        if (n === 2) && (saveinfo)

            #Paths generated for each method
            raw_inputs = [xP_oviedo, xP_queue, xP_random, xP_newton, xP_bfgs, xP_sgd]

            #Init the variables to set the limits of the sequence path
            g_xlims = (Inf, -Inf) 
            g_ylims = (Inf, -Inf)

            #Transform the matrices and update the global limits 
            processed_matrices = map(raw_inputs) do raw_data
                mat = Float64.(stack(raw_data, dims=1)) 
                
                #Update the global boundaries searching in each historial
                if !isempty(mat)
                    lx, ux = extrema(view(mat, :, 1))
                    g_xlims = (min(g_xlims[1], lx), max(g_xlims[2], ux))
                    ly, uy = extrema(view(mat, :, 2))
                    g_ylims = (min(g_ylims[1], ly), max(g_ylims[2], uy))
                end
                return mat
            end

            #Unpack the trajectories 
            xP_oviedo, xP_queue, xP_random, xP_newton, xP_bfgs, xP_sgd = processed_matrices

            #Visual path of the algorithms
            ax = plot_optimization_path(f, xP_oviedo, levels=20, label="Oviedo", g_xlims=g_xlims, g_ylims=g_ylims, function_name=name)

            #Add others paths in the plot
            add_optimization_path!(ax, xP_queue, label="Queue", color=:blue, linestyle=:dash)
            add_optimization_path!(ax, xP_newton, label="Newton", color=:green)
            add_optimization_path!(ax, xP_random, label="Random", color=:purple, linestyle=:dash)
            add_optimization_path!(ax, xP_bfgs, label="BFGS", color=:orange, linestyle=:dash)
            add_optimization_path!(ax, xP_sgd, label="SGD", color=:salmon, linestyle=:solid)

            #Save the plot
            savefig(ax, image_path)

        end
    else
        println(">>Normal mode")

        #Variables for the Random - NAMGM
        U = Any[0, 0.0, 0.0, 0.0, 0.0]
        M = vcat(fill(U', repetitions)...)

        #Execution of the NAMGM - methods and comparative methods
        xf_Oviedo, iterOviedo, t_oviedo, normOviedo, ttpSO, flagOviedo = namgmOviedo(f, g, h, x0, tol, nIters, modH, epsilonAdded, use_LS)
        xf_Queue, iterQueue, t_queue, normQueue, ttpSQ, flagQueue = namgmGrads(f, g, h, x0, tol, nIters, lqueue, modH, epsilonAdded, use_LS)
        for i in 1:repetitions
            M[i, :] .= namgmRandomVectors(f, g, h, x0, tol, nIters, randomsize, modH, epsilonAdded, show_info, use_LS)[2:end]
            U.+=M[i, :]
        end
        U./=repetitions
        iterRandom, t_random, normRandom, ttpSR, flagRandom = U
        solve_most_of_problems = (flagRandom >= 0.5) ? true : false
        displayResults("Random Mean ($repetitions repetitions)", iterRandom, t_random, normRandom, ttpSR, solve_most_of_problems)
        try
            xf_newton, iterNewton, t_newton, normNewton, ttpSN, flagNewton = newtonMethod(f, g, h, x0, tol, nIters, modN, epsilonAdded, use_LS)
        catch e
            println("An error was found in the execution of the Newton method: $e")
            xf_newton, iterNewton, t_newton, normNewton, ttpSN, flagNewton = x0, Inf, Inf, Inf, 0.0, false
        end    
        xf_bfgs, iterBFGS, t_bfgs, normBFGS, ttpSB, flagBFGS = BFGSMethod(f, g, h, x0, tol, nIters)
        xf_SGD, iterSGD, t_sgd, normSGD, ttpSGD, flagSGD = steepestMethod(f, g, x0, tol, nIters)

        #Create the frame for the random values
        random_df = DataFrame(M, headers)
        saveinfo ? CSV.write(file_name_results_historials, random_df) : nothing
        
    end

    #Remember to finalize the model when you are done
    finalize(nlp_problem)

    #Creation of Dataframe of the results
    times = [t_oviedo, t_queue, t_random, t_newton, t_bfgs, t_sgd]
    lastGrad = [normOviedo, normQueue, normRandom, normNewton, normBFGS, normSGD]
    iterations =[iterOviedo, iterQueue, iterRandom, iterNewton, iterBFGS, iterSGD]
    iterationsPerSecond =[ttpSO, ttpSQ, ttpSR, ttpSN, ttpSB, ttpSGD]
    convergence = [flagOviedo, flagQueue, flagRandom, flagNewton, flagBFGS, flagSGD] 
    data = [iterations, lastGrad, times, iterationsPerSecond, convergence]

    #Save the CSV file
    df = DataFrame(data, headers)

    #Write the CSV file
    if saveinfo
        CSV.write(file_name_results, df)
        !debug_mode ? CSV.write(file_name_results_historials, random_df) : nothing        
    end

    #Print the information of the written files
    println("-"^80)    
    println("Files written: ")
    if saveinfo
        println("- ",file_name_results)
        !debug_mode ? println("- ",file_name_results_historials) : nothing
        if debug_mode
            println("- ", file_name_historials)
            println("Images saved: \n", "- "*image_historial, "\n- "*image_conditionEvolution)
            n==2 ? println("- ", image_path) : nothing
        end
    else
        println("Not saveinfo flag was used, therefore any report has been saved.")
    end
       
    println("-"^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
