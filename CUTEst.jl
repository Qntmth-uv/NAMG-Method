using CUTEst, NLPModels

# Create a model for the 'ROSENBR' problem
nlp = CUTEstModel("ROSENBR")

# Access problem information and derivatives
println("x0 = $(nlp.meta.x0)")  # Initial point
fx = obj(nlp, nlp.meta.x0)      # Objective value at x0
gx = grad(nlp, nlp.meta.x0)     # Gradient at x0
Hx = hess(nlp, nlp.meta.x0)     # Hessian at x0
println("$fx  $gx  $Hx")


# Remember to finalize the model when you are done
finalize(nlp)