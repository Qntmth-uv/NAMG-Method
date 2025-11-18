# Converted from julia_notebook.ipynb
using LinearAlgebra
using Random
using Distributions
using Printf
using ProfileView
using Base: time
using CUTEst, NLPModels
using Plots
using DataStructures



function namgmSolver(Bk, gk, list_of_vectors)
    """Suponemos que el primer vector en la lista es necesariamente el vector del gradiente"""
    #Set the primal values
    n_rows = length(list_of_vectors)
    n = length(gk)
    matrix = zeros(n_rows, n_rows)
    V = [Bk * v for v in list_of_vectors]
    for i in (1:n_rows)
        for j in (i:n_rows) 
            matrix[i, j] = V[i]' * V[j]
            matrix[j, i] = matrix[i, j]
        end
    end
    b = [gk' * V[i] for i in (1:n_rows)]
    matrix = Symmetric(matrix)
    C = matrix\b
    return C
end 


function making_positive_foo_mod(m::Symmetric)

    # Get only the smallest and largest eigenvalues
    eigvals_extreme = eigvals(m, 1:1, size(m, 1):size(m, 1))
    eigen_min = eigvals_extreme[1]
    eigen_max = eigvals_extreme[end]

    if eigen_min < 0 && eigen_max < 0 
        eigen_max = abs(eigen_max)
        m += eigen_max*identity(m.shape[1]) 
    elseif eigen_min < 0
        eigen_min = abs(eigen_min)
        m += eigen_min*identity(m.shape[1])
    end
    distance = (eigen_max-eigen_min)/2

    return m, distance
end

function namgmOviedo(gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int)
    """NAMGM Using the Ovideo Directive. 
    # Input:
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
    
    # Output:
        -    x    : Vector - Final element of the sequence
        -  g_histo: Vector - Historial of gradient's thought the sequence 
    """
    #Start time
    start_time = time()
    gradient_his = []
    k = 1

    #Init the values
    x_old = x0
    g_old = gradient(x_old)

    #We make the first iteration to be able to compute sk and yk
    hg = Hessian(x_old) * g_old
    alpha = (g_old' * hg)/(hg' * hg)

    #Update the values in the first 
    v = -alpha * g_old
    x_new = x_old + v
    g_new = gradient(x_new)

    #Creation of the others elements in the set of vectors
    sk = x_new - x_old
    yk = g_new - g_old

    #Reasing the variables
    g_old = g_new
    x_old = x_new
    
    #Actual Norm
    gnorm = norm(g_old) 
    push!(gradient_his, gnorm)
    while (gnorm >= tolerance && k < maxIters)

        #Computing and making the hessian positive
        h = Symmetric(Hessian(x_old))
        #h, distance = making_positive_foo_mod(h)

        #Compute the coeficients
        V = [g_old, sk, yk]
        C = namgmSolver(h, g_old, V)

        #Update the squence and the set of vectors
        v = -sum(V .* C)
        x_new = x_old + v
        g_new = gradient(x_new)
        

        #Creation of the others elements in the set of vectors
        sk = x_new - x_old
        yk = g_new - g_old

        #Reasing the variables
        g_old = g_new
        x_old = x_new
        k+= 1

        #Push the gradients
        gnorm = norm(g_old) 
        push!(gradient_his, gnorm)
    end
    end_time = time()
    speended_time = end_time-start_time
    println("The last gradient was ", norm(g_old), ", iteration = ", k, ", time = ", speended_time, ".")

    return x_old, gradient_his
end

function namgmGrads(gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int, queue_size:: Int)
    """NAMGM Using the Gradient queue directive. 
    # Input:
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        - que_size:    Int   - Maximum number of elements in the queue (set of vectors).
    
    # Output:
        -    x    : Vector - Final element of the sequence
        -  g_histo: Vector - Historial of gradient's thought the sequence 
    """
    #Start time
    start_time = time()
    gradient_his = []
    gradient_queue = Queue{Vector}()
    k = 0
    
    #Init the variables
    x_old = x0
    gnorm = Inf
    while (k < maxIters && gnorm >= tolerance)
        #println("Iteration: ",k, " Queue size = ", length(gradient_queue))

        #Compute the gradient
        g_old = gradient(x_old)
        enqueue!(gradient_queue, g_old)

        #Dequee if it's needed
        if k >= queue_size
            dequeue!(gradient_queue)
        end

        #Compute the hessian
        h = Hessian(x_old)

        #Collect the current elements in the queue to solve the optimization problem
        V = collect(gradient_queue)
        C = namgmSolver(h, g_old, V)

        #Update the squence and the set of vectors
        v = -sum(V .* C)
        x_new = x_old + v
        g_new = gradient(x_new)
        
        #Reasing the variables
        g_old = g_new
        x_old = x_new

        #Update 
        gnorm = norm(g_old) 
        push!(gradient_his, gnorm)
        k+=1
    end
    end_time = time()
    speended_time = end_time-start_time
    println("The last gradient was ", gnorm, ", iteration = ", k, ", time = ", speended_time, ".")
    return x_old, gradient_his
end



function elementsTestFunction(name::String)
    """Function to use the function, gradient, Hessian of a fucntion, also
    return the x_minima and the nlp_problem to close it latter."""

    # Create a model for the 'name' problem
    nlp_problem = CUTEstModel(name)

    # Define callable functions
    function f(x)
        return obj(nlp_problem, x)
    end

    # Alternative gradient function that returns the gradient
    function g(x)
        return grad(nlp_problem, x)
    end

    # Alternative Hessian function
    function h(x)
        return hess(nlp_problem, x)
    end
    
    return f, g, h, nlp_problem.meta.x0, nlp_problem
end