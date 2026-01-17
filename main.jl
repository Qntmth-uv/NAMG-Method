using Random
using ArgParse


include("./NAMGM_methods.jl")
include("./hessian_mod.jl")
include("utils.jl")
include("plotting_util.jl")

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
            default=0
    end
    return parse_args(s)
end

#Mostrar el numero de condición
#implementar un Try-Catch
# Estudiar que pasa si le realizamos una modificacion en la digagonal y tridiagonal
# Ver si la implementacion del caso diagonal, tridiagonal, y desarrollar la solución exacta o implementarla, o usar un api.
# Ver que pasa con los algoritmos, porque son casi iguales al inicio.
# Medida de eficiencia de tiempo/iteraciones (promedio de iteraciones), o como promedio por iteracion.y promediar (guadar la media y la desviacion).
# Ver como se estan haciendo los calculos (por que existe la curiosidad de que los dos métodos inician igual y posteiormente se separan)
# Ver que pasa para un problema artificial bien condicionado (cuadratica, [100, 1000, 5000, 10000], con el objetivo 
# para ver si el numero de condicion influye en la convergencia del algoritmo.
# Ver porque es invariante a los vectores (habra alguna relacion entre los espacios generados) -


#Functions that take so much time: FMINSURF", NONMSQRT, "POWELLBS"
#Where Newton Fails: FLETCHCR, EXTROSNB

  function main()



    #Load the arguments
    parsed_args = parse_commandline()
    
    #Varoables
    problem = parsed_args["problem"]
    show_info = parsed_args["show_info"]
    image_name = parsed_args["image_name"]
    file_name = parsed_args["file_name"]
    nIters = parsed_args["nIters"]
    lqueue = parsed_args["lqueue"]
    tol = parsed_args["tol"]
    seed = parsed_args["seed"]
    
    #Fix a Seed for generation
    if seed == 0
        Random.seed!()
    else
        Random.seed!(seed) 
    end

    #Name of the results

    nombre = first(splitext(basename(problem)))
    if image_name == "gradient_evolution.png"
        image_name = "images/"*nombre*".png"
    end

    if file_name == "historial_results.csv"
        file_name = "csvs/"*nombre*".csv"
    end

    println(file_name, "  ",image_name)

    #Modifier Selection
    mod = lowercase(parsed_args["modifier"]) 
    if mod == "eigen"
        mod = modifyHessian_Eigen
    elseif mod == "diag"
        mod = diagonalModifier_Hessian
     elseif mod == "tridiag"
        mod = tridiagonalModifier_Hessian
    else
        mod = notModifierHessian
    end
    
    #Get the CUTEST functions    
    f, g, h, initial_point, nlp_problem = elementsTestFunction(problem)
    
    # Initialize random vector of same dimension
    n = length(initial_point)
    println("Dimension problem:", n)
    x0 = initial_point


    #Run the NAMGM algoritms
    xf_Oviedo, historial_Ovideo, t_oviedo, xP_oviedo, iterOviedo = namgmOviedo(g, h, x0, tol, nIters, mod)
    xf_Queue, historial_Queue, t_queue, xP_queue, iterQueue = namgmGrads(g, h, x0, tol, nIters, lqueue, mod)
    xf_random, historial_random, t_random, xP_random, iterRandom = namgmRandomVectors(g, h, x0, tol, nIters, lqueue, mod)
    xf_newton, historial_newton, t_newton, xP_newton, iterNewton = newtonMethod(g, h, x0, tol, nIters)

    #Clean the data if there is a 0.0 then the plot will explote.
    H = [historial_Ovideo, historial_Queue, historial_random, historial_newton]

    for i in 1:length(H)
        if H[i][end] == 0.0
           H[i][end] = 1e-15
        end
    end


    #Print the difference btw the given points
    println("Time per iteration (Oviedo): ", getIterationSpeed(historial_Ovideo, t_oviedo))
    println("Time per iteration (Queue): ", getIterationSpeed(historial_Queue, t_queue))
    println("Time per iteration (Random): ", getIterationSpeed(historial_random, t_random))
    println("Time per iteration (Newton): ", getIterationSpeed(historial_newton, t_newton))


    #Headers of the CSV file and creation 
    headers = ["Oviedo", "Queue", "RD", "Newton"]
    createCSV(H, file_name, headers)

    #Variables to plot the results
    problems_labels = ["Ovideo", "Queue", "RD", "Newton"]
    colors_list = [:blue, :red, :purple, :green]
    styles_list = [:solid, :dash, :dash, :solid]
    plotGradsEvolution(H, nombre, colors_list, problems_labels, styles_list, image_name)

    if n === 2
        #Transformation of trajectorys
        # 1. Agrupamos las listas crudas para iterar (principio DRY: Don't Repeat Yourself)
        raw_inputs = [xP_oviedo, xP_queue, xP_random, xP_newton]

        # 2. Inicializamos variables para los límites globales con infinito invertido
        g_xlims = (Inf, -Inf) # (min, max)
        g_ylims = (Inf, -Inf)

        # 3. Procesamos todo en un solo bucle (map) o comprensión
        # Convertimos a matrices y al mismo tiempo actualizamos los límites globales
        matrices_procesadas = map(raw_inputs) do raw_data
            # A. Conversión eficiente (stack es nativo y rápido en Julia >= 1.9)
            # Si usas una versión vieja, mantén: Float64.(reduce(hcat, raw_data)')
            mat = Float64.(stack(raw_data, dims=1)) 
            
            # B. Actualización de límites globales (usando extrema para rapidez)
            if !isempty(mat)
                # Eje X (Columna 1)
                lx, ux = extrema(view(mat, :, 1)) # view ahorra memoria
                g_xlims = (min(g_xlims[1], lx), max(g_xlims[2], ux))
                
                # Eje Y (Columna 2)
                ly, uy = extrema(view(mat, :, 2))
                g_ylims = (min(g_ylims[1], ly), max(g_ylims[2], uy))
            end
            
            return mat
        end

        # 4. Desempaquetamos los resultados (Asignación múltiple)
        xP_oviedo, xP_queue, xP_random, xP_newton = matrices_procesadas


        #Visual path of the algorithms
        mi_grafica = plot_optimization_path(f, xP_oviedo, levels=20, label="Oviedo", g_xlims=g_xlims, g_ylims=g_ylims, function_name=nombre)

        # # 3. ACTUALIZAR el lienzo con el segundo camino
        # # Pasamos 'mi_grafica' y cambiamos el color y etiqueta
        add_optimization_path!(mi_grafica, xP_queue, label="Queue", color=:blue, linestyle=:dash)
        add_optimization_path!(mi_grafica, xP_random, label="Random", color=:purple, linestyle=:dash)
        add_optimization_path!(mi_grafica, xP_newton, label="Newton", color=:green)


        #Guardar
        savefig(mi_grafica, "images/"*nombre*"_path.svg")
    end

    # Remember to finalize the model when you are done
    finalize(nlp_problem)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end