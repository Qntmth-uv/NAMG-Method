include("NAMGM_method.jl")

#Mostrar el numero de condición
#implementar un Try-Catch
# Estudiar que pasa si le realizamos una modificacion en la digagonal y tridiagonal
# Ver si la implementacion del caso diagonal, tridiagonal, y desarrollar la solución exacta o implementarla, o usar un api.

function main()
    problems = ["ROSENBR", "POWELLBS", "POWELLSG", "ARWHEAD", "BDQRTIC", 
            "BROYDN7D", "BRYBND", "CHNROSNB", "COSINE",
            "DIXMAANB", "DIXMAANC", "DIXMAAND", "DIXMAANF",
            "DIXMAANG", "DIXMAANH", "DIXMAANJ", "DIXMAANK",
            "DIXMAANL", "DQDRTIC", "EDENSCH", "ENGVAL1", "EXTROSNB",
            "FLETCBV2", "FLETCHCR", "FREUROTH", "GENROSE",
            "LIARWHD", "MOREBV", "NONCVXU2", "NONMSQRT", "PENALTY1",
            "PENALTY2", "PENALTY3", "POWER", "SROSENBR", "TESTQUAD",
            "TRIDIA", "VAREIGVL", "WOODS"]

    for p in problems[20:end]       
      actual_problem = p
      println(p)
      # FMINSURF"
      #Where Newton Fails: FLETCHCR, EXTROSNB
      
      f, g, h, solution_x, nlp_problem = elementsTestFunction(actual_problem)
      
      # Initialize random vector of same dimension
      n = length(solution_x)
      println("Dimension problem:", n)
      x0 = rand(n)

      #Run the NAMGM algoritms
      xf_Oviedo, historial_Ovideo = namgmOviedo(g, h, x0, 1.e-8, 10000)
      xf_Queue, historial_Queue = namgmGrads(g, h, x0, 1.e-8, 10000, 8)
      xf_newton, historial_newton = newtonMethod(g, h, x0, 1.e-8, 10000)
      
      #Print the difference btw the given points
      println("Distance btw obtained solution and original solution (Oviedo): ", norm(xf_Oviedo-solution_x))
      println("Distance btw obtained solution and original solution (Queue): ", norm(xf_Queue-solution_x))
      println("Distance btw obtained solution and original solution (Newton): ", norm(xf_newton-solution_x))
      
      #Plot the results
      plt = plot(historial_Ovideo,
          label="Oviedo - Gradient Norm",
          xlabel="Iteration",
          ylabel="‖∇f(x)‖",
          title="Convergence Analysis:"*p,
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
        linewidth=1,
        color=:red,
      )

      plot!(plt, historial_newton,
        label="Newton - Gradient Norm",
        linewidth=1,
        color=:green,
      )


      historial_newton
      
      savefig(plt, "gradient_evolution.png")

      # Remember to finalize the model when you are done
      finalize(nlp_problem)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end