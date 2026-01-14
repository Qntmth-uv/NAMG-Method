include("./NAMGM_method.jl")
include("./hessian_mod.jl")
include("utils.jl")
using ArgParse


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

# problems = ["ROSENBR", "POWELLSG", "BDQRTIC", 
#         "BROYDN7D", "BRYBND", "CHNROSNB", "COSINE",
#         "DIXMAANB", "DIXMAANC", "DIXMAAND", "DIXMAANF",
#         "DIXMAANG", "DIXMAANH", "DIXMAANJ", "DIXMAANK",
#         "DIXMAANL", "DQDRTIC", "EDENSCH", "ENGVAL1", "EXTROSNB",
#         "FLETCBV2", "FLETCHCR", "FREUROTH", "GENROSE",
#         "LIARWHD", "MOREBV", "NONCVXU2", "PENALTY1",
#         "PENALTY2", "PENALTY3", "POWER", "SROSENBR", "TESTQUAD",
#         "TRIDIA", "VAREIGVL", "WOODS"]

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
    f, g, h, solution_x, nlp_problem = elementsTestFunction(problem)
    
    # Initialize random vector of same dimension
    n = length(solution_x)
    println("Dimension problem:", n)
    x0 = rand(n)

    #Run the NAMGM algoritms
    xf_Oviedo, historial_Ovideo, t_oviedo = namgmOviedo(g, h, x0, tol, nIters, mod)
    xf_Queue, historial_Queue, t_queue = namgmGrads(g, h, x0, tol, nIters, lqueue, mod)
    xf_random, historial_random, t_random = namgmRandomVectors(g, h, x0, tol, nIters, lqueue, mod)
    xf_newton, historial_newton, t_newton = newtonMethod(g, h, x0, tol, nIters)

    #Clean the data if there is a 0.0 then the plot will explote.
    H = [historial_Ovideo, historial_Queue, historial_random, historial_newton]
    for i in 1:length(H)
        if H[i][end] == 0.0
           H[i][end] = 1e-15
        end

    end


    #Print the difference btw the given points
    println("Distance btw obtained solution and original solution (Oviedo): ", norm(xf_Oviedo-solution_x))
    println("Distance btw obtained solution and original solution (Queue): ", norm(xf_Queue-solution_x))
    println("Distance btw obtained solution and original solution (Random): ", norm(xf_random-solution_x))
    println("Distance btw obtained solution and original solution (Newton): ", norm(xf_newton-solution_x))

    #Print the difference btw the given points
    println("Time per iteration (Oviedo): ", getIterationSpeed(historial_Ovideo, t_oviedo))
    println("Time per iteration (Queue): ", getIterationSpeed(historial_Queue, t_queue))
    println("Time per iteration (Random): ", getIterationSpeed(historial_random, t_random))
    println("Time per iteration (Newton): ", getIterationSpeed(historial_newton, t_newton))



    #Headers of the CSV file
    headers = ["Oviedo", "Queue", "RD", "Newton"]
    #Creation of the  CSV file
    createCSV(H, file_name, headers)


    #Plot the results
    plt = plot(historial_Ovideo,
        label="Oviedo - Gradient Norm",
        xlabel="Iteration",
        ylabel="‖∇f(x)‖",
        title="Convergence Analysis:"*problem,
        linewidth=1,
        color=:blue,
        yscale=:log10,
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3,
        legend=:topright,
        size=(800, 400),
        dpi=300,
    )
    plot!(plt, historial_Queue,
      label="Queue - Gradient Norm",
      linestyle=:dash,
      linewidth=1,
      color=:red,
    )

    plot!(plt, historial_random,
      label="Random - Gradient Norm",
      linewidth=1,
      color=:yellow,
    )

    plot!(plt, historial_newton,
      label="Newton - Gradient Norm",
      linewidth=1,
      color=:green,
    )


    historial_newton
    
    savefig(plt, image_name)

    # Remember to finalize the model when you are done
    finalize(nlp_problem)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end