include("NAMGM_method.jl")

function main()
    problems = ["ROSENBR", "POWELLBS", "POWELLSG", "ARWHEAD", "BDQRTIC", 
            "BROYDN7D", "BRYBND", "CHNROSNB", "COSINE", "DIXMAANA",
            "DIXMAANB", "DIXMAANC", "DIXMAAND", "DIXMAANE", "DIXMAANF",
            "DIXMAANG", "DIXMAANH", "DIXMAANI", "DIXMAANJ", "DIXMAANK",
            "DIXMAANL", "DQDRTIC", "EDENSCH", "ENGVAL1", "EXTROSNB",
            "FLETCBV2", "FLETCHCR", "FMINSURF", "FREUROTH", "GENROSE",
            "LIARWHD", "MOREBV", "NONCVXU2", "NONMSQRT", "PENALTY1",
            "PENALTY2", "PENALTY3", "POWER", "SROSENBR", "TESTQUAD",
            "TRIDIA", "VAREIGVL", "WOODS"]

    actual_problem = "SROSENBR"
    f, g, h, solution_x, nlp_problem = elementsTestFunction(actual_problem)
    
    # Initialize random vector of same dimension
    n = length(solution_x)
    println("Dimension problem:", n)
    x0 = rand(n)

    #Run the NAMGM algoritms
    xf_Queue, historial_Queue = namgmGrads(g, h, x0, 1.e-8, 1000, 8)
    xf_Oviedo, historial_Ovideo = namgmOviedo(g, h, x0, 1.e-8, 1000)
    println("Distance btw obtained solution and original solution (Oviedo): ", norm(xf_Oviedo-solution_x))
    println("Distance btw obtained solution and original solution (Queue): ", norm(xf_Queue-solution_x))
    plt = plot(historial_Ovideo,
        label="Oviedo - Gradient Norm",
        xlabel="Iteration",
        ylabel="‖∇f(x)‖",
        title="Convergence Analysis",
        linewidth=3,
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
      linewidth=3,
      color=:red,
    )
    
    savefig(plt, "gradient_evolution.png")

    # Remember to finalize the model when you are done
    finalize(nlp_problem)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end