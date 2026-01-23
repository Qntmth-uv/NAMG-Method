"""
This is the same script that the NAMGM methods
only that this file contains other measurements needed
for the analysis. For example, this methods measure 
the condition number of the NAMGM system, the used
paths of the methods for low dimensions (bidimensional),
the gradient historial, and other standard measurements
like solution, number of iterations, executation time
"""


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
using Arpack



#Need to implement a BFGS method to compare.

function DEBUG_namgmSolver(Bk, gk, list_of_vectors)
    """Suponemos que el primer vector en la lista es necesariamente el vector del gradiente
    
    # Input
    
        - Bk: Matrix - Approximation of the hessian matrix.
        - gk: Vector - Gradient in the current iteration
        - lk: Array  - List of vectors to solve the system

    # Output:
        - C: Vector - Constans of the linear combination of the vectors
    """
    #Change the type of variable type of all cases to standarize the types
    list_of_vectors = [vec(Array(v)) for v in list_of_vectors]
    gk = vec(Array(gk)) 

    #Set the primal values
    n_rows = length(list_of_vectors)
    matrix = zeros(Float64, n_rows, n_rows)

    #Precompute the right part of the matrix coeficients and the vector b
    V = [Bk * v for v in list_of_vectors]
    
    #Construct the matrix system
    for i in (1:n_rows)
        for j in (i:n_rows) 
            matrix[i, j] = V[i]' * V[j]
            matrix[j, i] = matrix[i, j]
        end
    end

    #Construct the response vector
    b = [gk' * V[i] for i in (1:n_rows)]

    #Here we need a try-catch implementation, we can use a small modification of the hessian matrix
    matrix = Symmetric(matrix)

    #Compute the condition number of the matrix
    val_max = eigmax(matrix)
    val_min = eigmin(matrix)
    cn = abs(val_max/val_min)

    C=nothing
    #display(matrix)
    try
        C = matrix \ b
    catch
        matrix += 1e-5I
        C = matrix \ b
    end
    return C, cn
end 


function DEBUG_namgmOviedo(gradient::Function, Hessian::Function, x0:: Vector,
                           tolerance:: Float64, maxIters:: Int, hessian_mod::Function,
                           epsilon::Float64)
    """NAMGM Using the Ovideo Directive. 
    # Input:
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix.
    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence
    """
    #Start time
    start_time = time()
    gradient_his = []
    x_path = []
    condition_his = []
    dim = length(x0)
    k = 1

    #Steps matrix
    if dim == 2
        push!(x_path, x0)
    end

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
    
    #Steps matrix
    if dim == 2
        push!(x_path, x_new)
    end

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

        #Computing and making the hessian 
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, g_new, epsilon)
 
        #Compute the coeficients
        V = [g_old, sk, yk]
        C, cn = DEBUG_namgmSolver(h, g_old, V)

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
        push!(condition_his, cn)
         #Steps matrix
        if dim == 2
            push!(x_path, x_old)
        end
    end
    end_time = time()
    speended_time = end_time-start_time
    println("The last gradient was ", norm(g_old), ", iteration = ", k, ", time = ", speended_time, ".")

    return x_old, gradient_his, speended_time, x_path, k, gnorm, condition_his
end

function DEBUG_namgmGrads(gradient::Function, Hessian::Function, x0:: Vector, 
                          tolerance:: Float64, maxIters:: Int, queue_size:: Int, 
                          hessian_mod::Function, epsilon::Float64)
    """NAMGM Using the Gradient queue directive. 
    # Input:
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        - que_size:    Int   - Maximum number of elements in the queue (set of vectors).
        -hessian_mod: Call   - Modifier of the hessian matrix.
    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence
    """
    #Start time
    start_time = time()
    x_old = x0
    
    gradient_his = []
    x_path = []
    condition_his = []
    gradient_queue = Queue{Vector}()
    k = 0
    dim = length(x0)
    if dim == 2
        push!(x_path, x0)
    end
    #Init the variables
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
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, g_old, epsilon)
        
        #Collect the current elements in the queue to solve the optimization problem
        V = collect(gradient_queue)
        C, cn = DEBUG_namgmSolver(h, g_old, V)

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
        push!(condition_his, cn)
        if dim == 2
            push!(x_path, x_old)
        end
        k+=1
    end
    
    #Finalization of the method
    end_time = time()
    speended_time = end_time-start_time

    #Print report of solutions
    println("The last gradient was ", gnorm, ", iteration = ", k, ", time = ", speended_time, ".")
    return x_old, gradient_his, speended_time, x_path, k, gnorm, condition_his
end


function DEBUG_namgmRandomVectors(gradient::Function, Hessian::Function, x0:: Vector, 
                                  tolerance:: Float64, maxIters:: Int, randomSize:: Int, 
                                  hessian_mod::Function, epsilon::Float64)
    """NAMGM Using the Random Vectors directive. 
    #Input
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        -randomSize:  Int.   - Number of random elements to take
        -hessian_mod: Call   - Modifier of the hessian matrix
    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence
    """
    #Start time
    x_old = x0
    dim = length(x0)
    start_time = time()
    gradient_his = []
    condition_his = []
    x_path = []
    k = 0

    #If the dimension of the objetive function is 2 then we save the path
    if dim == 2
        push!(x_path, x0)
    end

    #Init the variables
    gnorm = Inf
    
    #Initializate the sequcence generation
    while (k < maxIters && gnorm >= tolerance)
        #Compute the gradient
        g_old = gradient(x_old)

        #Initialize an empty list specifically for Vectors
        V = Vector{Vector{Float64}}()
        vectorsSample = randn(Float64, randomSize, dim)

        #Add g_old as the FIRST whole vector
        push!(V, vec(Array(g_old)))

        #We loop through rows and push them one by one
        for i in 1:size(vectorsSample, 1)
            push!(V, vec(vectorsSample[i, :]))
        end

        #Get the approximation of the hessian
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, g_old, epsilon)
  
        #Collect the currect elements in the queue to solve the optimization problem
        C, cn = DEBUG_namgmSolver(h, g_old, V)

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
        push!(condition_his, cn)
        if dim == 2
            push!(x_path, x_old)
        end
        k+=1
    end

    #Finalization of the method
    end_time = time()
    speended_time = end_time-start_time

    #Print report of solution
    println("The last gradient was ", gnorm, ", iteration = ", k, ", time = ", speended_time, ".")
    return x_old, gradient_his, speended_time, x_path, k, gnorm, condition_his
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

function DEBUG_newtonMethod(gradient::Function, hessian::Function, 
                            x0::Vector, tolerance::Float64, maxIters:: Int)
    """
    Newton's method with applicable modifier in the Hessian matrix
    (This is not the same as the BFGS method, such method use a unique modifer)
    # Input:
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix.

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence    
    """
    #Start time of the method
    start_time = time()
    gradient_his = []
    x_path = []
    dim = length(x0)
    if dim == 2
        push!(x_path, x0)
    end
    k = 0

    #Init the variables
    x = x0
    gnorm = Inf
    while (k < maxIters && gnorm >= tolerance)
        #Compute the gradient of the matrix
        gk = gradient(x)
        
        #Implemented Try catch in case that the hessian it's not invertible.
        hk = Symmetric(hessian(x))
        s = nothing
        try
            s = -hk\gk    
        catch
            s = -1e-3gk
            println("Used Fixed gradient descent")
        end
        x = x+s

        #Update 
        gnorm = norm(gk) 
        push!(gradient_his, gnorm)
        if dim == 2
            push!(x_path, x)
        end
        k+=1
    end
    #Finalization of the method
    end_time = time()
    speended_time = end_time-start_time

    #Print report of solutions
    println("The last gradient was ", gnorm, ", iteration = ", k, ", time = ", speended_time, ".")
    return x, gradient_his, speended_time, x_path, k, gnorm
end