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

function BB_Aproximattion(s::Vector,y::Vector)
    a = (s' * s)/(s' * y)
    I_n = Matrix{Float32}(I, length(s), length(s))
    return a*I_n
end


function fasCosineSim(u::Vector, v::Vector)
    """Computes the cosine similarity between two vectors. In an efficient way
    """
    dot_prod = 0.0
    norm_u = 0.0
    norm_v = 0.0
    @simd for i in eachindex(u, v)
        dot_prod += u[i] * v[i]
        norm_u += u[i]^2
        norm_v += v[i]^2
    end
    return dot_prod / (sqrt(norm_u) * sqrt(norm_v))
end



function diagonalApproximation(hessianMatrix::Symmetric, epsion::Float64 = 1e-12)
    """Fucntion that gets the main diagonal of the Hessian Matrix plus an epsilon- if it's
    needed.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Outpu:
        -h: Diaognal - The main diagonal of the approximation of the hesssian.        
        """
    h = Diagonal(diag(hessianMatrix)) + epsion * I
    return h
end
   

function making_positive_foo_mod(m::Symmetric)
    n = size(m, 1)
    
    # Use eigs for extreme eigenvalues of sparse matrices
    eigen_min = eigs(m, which=:SR, nev=1, ritzvec=false)[1][1]  # Smallest real
    eigen_max = eigs(m, which=:LR, nev=1, ritzvec=false)[1][1]  # Largest real
    
    if eigen_min < 0 && eigen_max < 0 
        shift = abs(eigen_max)
        m_modified = m + shift * I
    elseif eigen_min < 0
        shift = abs(eigen_min)
        m_modified = m + shift * I
    else
        m_modified = m
        shift = 0.0
    end
    
    distance = (eigen_max - eigen_min) / 2
    
    return m_modified, distance
end

function namgmSolver(Bk, gk, list_of_vectors)
    """Suponemos que el primer vector en la lista es necesariamente el vector del gradiente
    
    Parece ser q
    
    
    """
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

        #Computing and making the hessian 
        #h = Symmetric(BB_Aproximattion(sk, yk))
        #h, distance = making_positive_foo_mod(h)
        h = Symmetric(Hessian(x_old))
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
    x_old = x0
    start_time = time()
    gradient_his = []
    gradient_queue = Queue{Vector}()
    k = 0
    
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
        h = Hessian(x_old)
        #try
        #    F = Cholesky(h)
        #catch e
        #    println("Error caught: ")
        #end
        #Using diagonal of the hessian
        #h = Diagonal(diag(h)) + 1e-12 * I

        #Using the tridiagonal of the hessian
        #h = Tridiagonal(h) + 1e-12 * I

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
        k+=1
    end
    
    #Finalization of the method
    end_time = time()
    speended_time = end_time-start_time

    #Print report of solutions
    println("The last gradient was ", gnorm, ", iteration = ", k, ", time = ", speended_time, ".")
    return x_old, gradient_his
end


function namgmRandomVectors(gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int, randomSize:: Int)
    """NAMGM Using the Random Vectors directive. """
    #Start time
    x_old = x0
    dim = length(x0)
    start_time = time()
    gradient_his = []
    k = 0

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
        h = Hessian(x_old)

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
        k+=1
    end

    #Finalization of the method
    end_time = time()
    speended_time = end_time-start_time

    #Print report of solutions
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

function newtonMethod(gradient, hessian, x0::Vector, tolerance::Float64, maxIters:: Int)
    #Start time of the method
    start_time = time()
    gradient_his = []
    k = 0

    #Init the variables
    x = x0
    gnorm = Inf
    while (k < maxIters && gnorm >= tolerance)
        gk = gradient(x)
        
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
        k+=1
    end
    #Finalization of the method
    end_time = time()
    speended_time = end_time-start_time

    #Print report of solutions
    println("The last gradient was ", gnorm, ", iteration = ", k, ", time = ", speended_time, ".")
    return x, gradient_his
end