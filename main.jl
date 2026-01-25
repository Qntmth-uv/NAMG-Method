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
        "--lqueue"
            help = "Maximum number of elements in the NAMGMGradQueue"
            arg_type = Int
            default = 8
        "--tol"
            help = "Minimum aceptable gradient norm"
            arg_type = Float64
            default = 1.e-8
        "--modifierH"
            help = "Hessian modifier Strategy"
            arg_type = String
            default = "eigen"
        "--modifierS"
            help = "System modifier Strategy"
            arg_type = String
            default = "none"
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
        "--useDimProblem"
            help= "Use the same number of vectors as the dimension of the problem (valid only on RandomVectors)"
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
    usedim_problem = parsed_args["useDimProblem"] 
    image_name = parsed_args["image_name"]
    file_name = parsed_args["file_name"]
    nIters = parsed_args["nIters"]
    lqueue = parsed_args["lqueue"]
    randomsize = lqueue
    tol = parsed_args["tol"]
    seed = parsed_args["seed"]
    epsilonAdded = parsed_args["epsilon"]
    modH = lowercase(parsed_args["modifierH"])
    modS = lowercase(parsed_args["modifierS"])

    #Fix a Seed for generation
    seed == 0 ? Random.seed!() : Random.seed!(seed)

    #Name of the results
    name = first(splitext(basename(problem)))
    if image_name == "default"
        hitorialName = "images/historials_grads/"*name*"_"*modS*".png"
        pathName = "images/paths/"*name*"_"*modS*"_path.svg"
        cnName ="images/condition_analysis/"*name*"_"*modS*".png" 
    end
    if file_name == "default"
        file_name_results = "csvs/results/"*name*"_"*modS*".csv"
        file_name_historials = "csvs/historials/"*name*"_"*modS*".csv"
    end
    
    #Pritn thte values of the parser
    println("-"^40)
    @printf("%-20s | %-20s\n", "Parameters", "Value")
    println("-"^40)

    # 'sort' convierte el Dict en un vector de pares ordenados por la llave
    for (key) in sort(collect(keys(parsed_args)))
        @printf("%-20s | %-20s\n", key, parsed_args[key])
    end
    println("-"^40)


    #println("Modifier Hessian: ", modH, " | Modifier System: ", modS)
    modH = get_modifier(modH)
    modS = get_modifier(modS)

    #Get the CUTEST functions    
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem)
    
    # Initialize random vector of same dimension
    n = length(initial_point)
    println("Dimension of the problem: ", n)
    x0 = initial_point


    #Check the flag of the size of RandomVectors
    usedim_problem ? randomsize = n : nothing
    println("Random Vectors being used: ", randomsize-1)
    println("-"^40)
    if debug_mode
        println("DEBUG Mode")

        #Run the NAMGM algoritms using DEBUG mesuarements
        xf_Oviedo, historial_Ovideo, t_oviedo, xP_oviedo, iterOviedo, normOviedo, CN_O, ttpSO = DEBUG_namgmOviedo(g, h, x0, tol, nIters, modH, epsilonAdded, modS)
        xf_Queue, historial_Queue, t_queue, xP_queue, iterQueue, normQueue, CN_Q, ttpSQ = DEBUG_namgmGrads(g, h, x0, tol, nIters, lqueue, modH, epsilonAdded, modS)
        xf_random, historial_random, t_random, xP_random, iterRandom, normRandom, CN_R, ttpSR = DEBUG_namgmRandomVectors(g, h, x0, tol, nIters, randomsize, modH, epsilonAdded, modS)
        xf_newton, historial_newton, t_newton, xP_newton, iterNewton, normNewton, ttpSN = DEBUG_newtonMethod(g, h, x0, tol, nIters)
        xf_bfgs, historial_bfgs, t_bfgs, xP_bfgs, iterBFGS, normBFGS, ttpSB = DEBUG_BFGSMethod(f, g, h, x0, tol, nIters)

        #Clean the data if there is a 0.0 then the plot will explote.
        H = [historial_Ovideo, historial_Queue, historial_random, historial_newton, historial_bfgs]
        Hc = [CN_O, CN_Q, CN_R]
        for i in 1:length(H)
            if H[i][end] == 0.0 || isnan(H[i][end])
                H[i][end] = 0.0
            end
        end

        #Headers of the CSV file and creation 
        headers = ["Oviedo", "Queue", "RD", "Newton", "BFGS"]
        createCSV(H, file_name_historials, headers)

        #Variables to plot the results
        problems_labels = ["Ovideo", "Queue", "RD", "Newton", "BFGS"]
        colors_list = [:blue, :red, :purple, :green, :orange]
        styles_list = [:solid, :dash, :dash, :solid, :dash]

        #Plot the results (The gradient historial and the condition number)
        plot_title = "Convergence Analysis: " * name
        plot_titleCN = "Condition Analysis: " * name
        plotEvolution(H, name, colors_list, problems_labels, styles_list, "||∇f(x)||", plot_title, hitorialName)
        plotEvolution(Hc, name, colors_list[1:end-1], problems_labels[1:end-1], styles_list[1:end-1], "κ(Hψ)", plot_titleCN, cnName)

        #If the function is bidimensioal, then plot the sequence path
        if n === 2
            #Transformation of trajectorys
            raw_inputs = [xP_oviedo, xP_queue, xP_random, xP_newton, xP_bfgs]

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
            xP_oviedo, xP_queue, xP_random, xP_newton, xP_bfgs = matrices_procesadas

            #Visual path of the algorithms
            ax = plot_optimization_path(f, xP_oviedo, levels=20, label="Oviedo", g_xlims=g_xlims, g_ylims=g_ylims, function_name=name)

            #Add others paths in the plot
            add_optimization_path!(ax, xP_queue, label="Queue", color=:blue, linestyle=:dash)
            add_optimization_path!(ax, xP_newton, label="Newton", color=:green)
            add_optimization_path!(ax, xP_random, label="Random", color=:purple, linestyle=:dash)
            add_optimization_path!(ax, xP_bfgs, label="BFGS", color=:orange, linestyle=:dash)

            #Save the plot
            savefig(ax, pathName)
        end
    else
        println("Normal mode")
        xf_Oviedo, iterOviedo, t_oviedo, normOviedo, ttpSO = namgmOviedo(g, h, x0, tol, nIters, modH, epsilonAdded)
        xf_Queue, iterQueue, t_queue, normQueue, ttpSQ = namgmGrads(g, h, x0, tol, nIters, lqueue, modH, epsilonAdded)
        xf_random, iterRandom, t_random, normRandom, ttpSR = namgmRandomVectors(g, h, x0, tol, nIters, randomsize, modH, epsilonAdded)
        xf_newton, iterNewton, t_newton, normNewton, ttpSN = newtonMethod(g, h, x0, tol, nIters)
        xf_bfgs, iterBFGS, t_bfgs, normBFGS, ttpSB = BFGSMethod(f, g, h, x0, tol, nIters)
    end

    # Remember to finalize the model when you are done
    finalize(nlp_problem)


    #Creation of Dataframe of the results
    times = [t_oviedo, t_queue, t_random, t_newton, t_bfgs]
    lastGrad = [normOviedo, normQueue, normRandom, normNewton, normBFGS]
    iterations =[iterOviedo, iterQueue, iterRandom, iterNewton, iterBFGS]
    iterationsPerSecond =[ttpSO, ttpSQ, ttpSR, ttpSN, ttpSB]
    data = [iterations, lastGrad, times, iterationsPerSecond]

    #Header of the DF
    headers = ["iterations", "Last Gradient", "Execution time", "Iterations per Second"]

    #Save the CSV file
    df = DataFrame(data, headers)

    #Write the CSV file
    CSV.write(file_name_results, df)
    print("-"^40)
    #Print of information written
    println("\nFiles writed: ")
    println("- ",file_name_results)
    if debug_mode
        println("- ", file_name_historials)
        println("Images saved: \n", "- "*hitorialName, "\n- "*cnName)
        n==2 ? println("- ", pathName) : nothing
    end
    println("-"^40)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end