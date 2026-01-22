using Random
using ArgParse


"""

Execution command (example)
julia --project=venv_NAMGM main.jl --problem cutest-sif/ROSENBR.SIF --nIters 5 --modifier eigen 
julia --project=venv_NAMGM main.jl --problem cutest-sif/ROSENBR.SIF --modifier eigen 

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
            help = "Show info of the problem a boolean flag (true if present)"
            action = :store_true
        "--image_name"
            help = "Name of the result plot"
            arg_type = String
            default = "gradient_evolution.png"
        "--file_name"
            help = "Name of the output file (CSV)"
            arg_type = String
            default = "historial_results.csv"
        "--nIters"
            help = "Maximum number of iterations of the method"
            arg_type = Int
            default = 100
        "--lqueue"
            help = "Maximum number of elements in the NAMGMGradQueue"
            arg_type = Int
            default = 8
        "--tol"
            help = "Minimum aceptable gradient norm"
            arg_type = Float64
            default = 1.e-8
        "--modifier"
            help = "Hessian modifier Strategy"
            arg_type = String
            default = "eigen"
        "--seed"
            help = "Fix a seed for the random process"
            arg_type = Int
            default = 0
        "--epsilon"
            help = "Epsion added to the Modfier in case of being needed"
            arg_type = Float64
            default = minValue
        "--DEBUG"
            help= "Use the DEBUG functions of the methos to create CSV's and some informative Plots"
            action = :store_true
    end
    return parse_args(s)
end


  function main()

    #Load the arguments
    parsed_args = parse_commandline()
    
    #Definied variables
    problem = parsed_args["problem"]
    show_info = parsed_args["show_info"]
    debug_mode = parsed_args["DEBUG"]
    image_name = parsed_args["image_name"]
    file_name = parsed_args["file_name"]
    nIters = parsed_args["nIters"]
    lqueue = parsed_args["lqueue"]
    tol = parsed_args["tol"]
    seed = parsed_args["seed"]
    epsilonAdded = parsed_args["epsilon"]
    mod = lowercase(parsed_args["modifier"]) 
    
    #Fix a Seed for generation
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Name of the results
    name = first(splitext(basename(problem)))
    if image_name == "gradient_evolution.png"
        image_name = "images/"*name*".png"
        cnName ="images/CNA_"*mod*"_"*name*".png" 
    end
    if file_name == "historial_results.csv"
        file_name = "csvs/"*name*".csv"
    end
    println(file_name, "  ",image_name)

    #Modifier Selection
    if mod == "eigen"
        mod = modifyHessian_Eigen
    elseif mod == "diag"
        mod = diagonalModifier_Hessian
     elseif mod == "tridiag"
        mod = tridiagonalModifier_Hessian
     elseif mod == "remove"
        mod = removeConvergenceModifier
    else
        mod = notModifierHessian
    end
    
    #Get the CUTEST functions    
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem)
    
    # Initialize random vector of same dimension
    n = length(initial_point)
    println("Dimension problem:", n)
    x0 = initial_point


    if debug_mode
        println("DEBUG Mode")

        #Run the NAMGM algoritms using DEBUG mesuarements
        xf_Oviedo, historial_Ovideo, t_oviedo, xP_oviedo, iterOviedo, CN_O = DEBUG_namgmOviedo(g, h, x0, tol, nIters, mod, epsilonAdded)
        xf_Queue, historial_Queue, t_queue, xP_queue, iterQueue, CN_Q = DEBUG_namgmGrads(g, h, x0, tol, nIters, lqueue, mod, epsilonAdded)
        xf_random, historial_random, t_random, xP_random, iterRandom, CN_R= DEBUG_namgmRandomVectors(g, h, x0, tol, nIters, lqueue, mod, epsilonAdded)
        xf_newton, historial_newton, t_newton, xP_newton, iterNewton = DEBUG_newtonMethod(g, h, x0, tol, nIters)

        #Clean the data if there is a 0.0 then the plot will explote.
        H = [historial_Ovideo, historial_Queue, historial_random, historial_newton]
        Hc = [CN_O, CN_Q, CN_R]
        for i in 1:length(H)
            if H[i][end] == 0.0
            H[i][end] = minValue
            end
        end

        #Headers of the CSV file and creation 
        headers = ["Oviedo", "Queue", "RD", "Newton"]
        createCSV(H, file_name, headers)

        #Variables to plot the results
        problems_labels = ["Ovideo", "Queue", "RD", "Newton"]
        colors_list = [:blue, :red, :purple, :green]
        styles_list = [:solid, :dash, :dash, :solid]

        #Plot the results (The gradient historial and the condition number)
        plotGradsEvolution(H, name, colors_list, problems_labels, styles_list, image_name)
        plotConditionEvolution(Hc, name, colors_list[1:end-1], problems_labels[1:end-1], styles_list[1:end-1], cnName)

        #If the function is bidimensioal, then plot the sequence path
        if n === 2
            #Transformation of trajectorys
            raw_inputs = [xP_oviedo, xP_queue, xP_random, xP_newton]

            #Init the variables to set the limits of the sequence path
            g_xlims = (Inf, -Inf) 
            g_ylims = (Inf, -Inf)

            #Transform the matrices and update the global limits 
            matrices_procesadas = map(raw_inputs) do raw_data
                mat = Float64.(stack(raw_data, dims=1)) 
                
                #Update the global boundries searching in each historial
                if !isempty(mat)
                    lx, ux = extrema(view(mat, :, 1))
                    g_xlims = (min(g_xlims[1], lx), max(g_xlims[2], ux))
                    ly, uy = extrema(view(mat, :, 2))
                    g_ylims = (min(g_ylims[1], ly), max(g_ylims[2], uy))
                end
                return mat
            end

            #Unpack the trajectorys 
            xP_oviedo, xP_queue, xP_random, xP_newton = matrices_procesadas

            #Visual path of the algorithms
            ax = plot_optimization_path(f, xP_oviedo, levels=20, label="Oviedo", g_xlims=g_xlims, g_ylims=g_ylims, function_name=name)

            #Add others paths in the plot
            add_optimization_path!(ax, xP_queue, label="Queue", color=:blue, linestyle=:dash)
            add_optimization_path!(ax, xP_newton, label="Newton", color=:green)
            add_optimization_path!(ax, xP_random, label="Random", color=:purple, linestyle=:dash)

            #Save the plot
            savefig(ax, "images/"*name*"_path.svg")
        end
    else
        println("Normal mode")
        xf_Oviedo, iterOviedo, t_oviedo, normOviedo = DEBUG_namgmOviedo(g, h, x0, tol, nIters, mod, epsilonAdded)
        xf_Queue, iterQueue, t_queue, normQueue = DEBUG_namgmGrads(g, h, x0, tol, nIters, lqueue, mod, epsilonAdded)
        xf_random, iterRandom, t_random, normRandom = DEBUG_namgmRandomVectors(g, h, x0, tol, nIters, lqueue, mod, epsilonAdded)
        xf_newton, iterNewton, t_newton, normNewton = DEBUG_newtonMethod(g, h, x0, tol, nIters)
    end

    #Print the number of iterations per second
    println("Time per iteration (Oviedo): ", getIterationSpeed(historial_Ovideo, t_oviedo))
    println("Time per iteration (Queue): ", getIterationSpeed(historial_Queue, t_queue))
    println("Time per iteration (Random): ", getIterationSpeed(historial_random, t_random))
    println("Time per iteration (Newton): ", getIterationSpeed(historial_newton, t_newton))

    # Remember to finalize the model when you are done
    finalize(nlp_problem)

    #Creation of Dataframe of the results
    times = [t_oviedo, t_queue, t_random, t_newton]
    lastGrad = [historial_Ovideo[end], historial_Queue[end], historial_random[end], historial_newton[end]]
    iterations =[iterOviedo, iterQueue, iterRandom, iterNewton]
    data = [iterations, lastGrad, times]

    #Header of the DF
    headers = ["iterations", "Last Gradient", "Execution time"]

    #Save the CSV file
    df = DataFrame(data, headers)

    #Write the CSV file
    CSV.write("csvs/results_"*name*".csv", df)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end