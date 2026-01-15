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


function namgmSolver(Bk, gk, list_of_vectors)
    """Suponemos que el primer vector en la lista es necesariamente el vector del gradiente
    
    Parece ser q
    
    
    """
    #Set the primal values
    n_rows = length(list_of_vectors)
    matrix = zeros(n_rows, n_rows)
    V = [Bk * v for v in list_of_vectors]
    for i in (1:n_rows)
        for j in (i:n_rows) 
            matrix[i, j] = V[i]' * V[j]
            matrix[j, i] = matrix[i, j]
        end
    end
    b = [gk' * V[i] for i in (1:n_rows)]

    #Here we need a try-catch implementation, we can use a small modification of the hessian matrix
    matrix = Symmetric(matrix)
    C=nothing
    #display(matrix)
    try
        C = matrix \ b
    catch
        matrix += 1e-5I
        C = matrix \ b
    end
    return C
end 

function namgmSolver_V2(Bk, gk, list_of_vectors)
    """
    Solves the optimization problem, ensuring all inputs are handled as dense vectors.
    """
    
    # --- STEP 1: SANITIZATION (The Fix) ---
    # We create a new clean list where every element is forced to be a dense Vector{Float64}.
    # vec(Array(v)) handles SparseMatrices, 1xN matrices, and standard vectors.
    clean_vectors = [vec(Array(v)) for v in list_of_vectors]
    
    # Ensure gk is also a dense vector
    gk_clean = vec(Array(gk))

    # --- STEP 2: PRE-CALCULATION ---
    n_rows = length(clean_vectors)
    
    # Compute V = Bk * v for every vector. 
    # We assume Bk is the Hessian. The result is stored as a list of vectors.
    V = [Bk * v for v in clean_vectors]

    # --- STEP 3: BUILD THE MATRIX ---
    matrix = zeros(Float64, n_rows, n_rows) # Explicitly use Float64

    for i in 1:n_rows
        # Optimization: Matrix is symmetric, so only compute j >= i
        for j in i:n_rows 
            # Dot product of the pre-computed vectors
            val = dot(V[i], V[j]) 
            matrix[i, j] = val
            matrix[j, i] = val
        end
    end

    # --- STEP 4: COMPUTE RHS (b) AND SOLVE ---
    # b[i] = gk' * V[i]
    b = [dot(gk_clean, V[i]) for i in 1:n_rows]

    # Use Symmetric wrapper to tell the solver to use a faster Cholesky/LDLT factorization
    # standard 'matrix \ b' is robust
    C = Symmetric(matrix) \ b
    
    return C
end

function namgmOviedo(gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int, hessian_mod)
    """NAMGM Using the Ovideo Directive. 
    # Input:
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix.
    # Output:
        -    x    : Vector - Final element of the sequence
        -  g_histo: Vector - Historial of gradient's thought the sequence 
    """
    #Start time
    start_time = time()
    gradient_his = []
    x_path = []
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
        #h = Symmetric(BB_Aproximattion(sk, yk))
        #h, distance = making_positive_foo_mod(h)
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, dim)

        #h = Diagonal(diag(h)) + 1e-12 * I
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
         #Steps matrix
        if dim == 2
            push!(x_path, x_old)
        end
    end
    end_time = time()
    speended_time = end_time-start_time
    println("The last gradient was ", norm(g_old), ", iteration = ", k, ", time = ", speended_time, ".")

    return x_old, gradient_his, speended_time, x_path
end

function namgmGrads(gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int, queue_size:: Int, hessian_mod)
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
        -    x    : Vector - Final element of the sequence
        -  g_histo: Vector - Historial of gradient's thought the sequence 
    """
    #Start time
    x_old = x0
    start_time = time()
    gradient_his = []
    x_path = []
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
        enqueue!(gradient_queue, g_old + 1e-4 * randn(dim))

        #Dequee if it's needed
        if k >= queue_size
            dequeue!(gradient_queue)
        end

        #Compute the hessian
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, dim)
  

        #println("Iteracion: ",k," Condicion: ",cond(Matrix(h)))

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
    return x_old, gradient_his, speended_time, x_path
end


function namgmRandomVectors(gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int, randomSize:: Int, hessian_mod)
    """NAMGM Using the Random Vectors directive. 
    
    #Input
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the fucnion.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        -randomSize:  Int.   - Number of random elements to take
        -hessian_mod: Call   - Modifier of the hessian matrix
    # Output:
        -    x    : Vector - Final element of the sequence
        -  g_histo: Vector - Historial of gradient's thought the sequence 
    """
    #Start time
    x_old = x0
    dim = length(x0)
    start_time = time()
    gradient_his = []
    x_path = []
    k = 0
    if dim == 2
        push!(x_path, x0)
    end
    #Init the variables
    gnorm = Inf
    
    while (k < maxIters && gnorm >= tolerance)
        #Compute the gradient
        g_old = gradient(x_old)

        # 1. Initialize an empty list specifically for Vectors
        V = Vector{Vector{Float64}}()
        vectorsSample = 1000*randn(Float64, randomSize, dim)
        # 2. Add g_old as the FIRST whole vector
        push!(V, vec(Array(g_old)))

        # 3. Append the random vectors
        # We loop through rows and push them one by one
        for i in 1:size(vectorsSample, 1)
            push!(V, vec(vectorsSample[i, :]))
        end

        #Get the approximation of the hessian
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, dim)
  
        #Collect the currect elements in the queue to solve the optimization problem
        C = namgmSolver_V2(h, g_old, V)

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
    return x_old, gradient_his, speended_time, x_path
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

function newtonMethod(gradient, hessian, x0::Vector, tolerance::Float64, maxIters:: Int)
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
    return x, gradient_his, speended_time, x_path
end