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

#Add tools
include("utils.jl")

function namgmSolver(Bk, gk, list_of_vectors)
    """Function to solve the optimization problem to find the factors in the linear combination
    of the given vectors (list_of_vectors). This method does not allow the modification of Hessian
    of the matrix. The DEBUG mode allows it, and other type of modification like BFGS should be implemented
    in other function and file.
    
    # Input
        - Bk: Matrix - Approximation of the hessian matrix.
        - gk: Vector - Gradient in the current iteration
        - lk: Array  - List of vectors to solve the system

    # Output:
        - C: Vector - Constants of the linear combination of the vectors
    """
     #Change the type of variable type of all cases to standardize the types
    list_of_vectors = [vec(Array(v)) for v in list_of_vectors]
    gk = vec(Array(gk)) 

    #Set the primal values
    n_rows = length(list_of_vectors)
    matrix = zeros(Float64, n_rows, n_rows)

    #Precompute the right part of the matrix coefficients and the vector b
    V = [Bk * v for v in list_of_vectors]
    
    #Construct the matrix system
    for i in (1:n_rows)
        for j in (i:n_rows) 
            matrix[i, j] = dot(V[i], V[j])
            matrix[j, i] = matrix[i, j]
        end
    end

    #Construct the response vector
    b = [dot(gk, V[i]) for i in (1:n_rows)]

    #Here we need a try-catch implementation, we can use a small modification of the hessian matrix
    matrix = Symmetric(matrix)
    C=nothing
    try
        C = matrix \ b
    catch
        C = namgmSolver(Bk, gk, list_of_vectors[1:end-1])
        C = [i <= length(C) ? C[i] : 0.0 for i in 1:length(list_of_vectors)]
    end
    return C
end 

function namgmOviedo(fx::Function, gradient, Hessian, x0:: Vector, tolerance:: Float64, 
                    maxIters:: Int, hessian_mod, epsilon::Float64, add_lineSearch::Bool = false,
                    show_results::Bool = false)
    """The NAMGM algorithm using the set of vectors in the paper AMGM algorithm [1]. There are other authors, but
    to maintain simple and memorable the name, we use only the first name of the main author.
    # Input:
        -    fx   : Function  - Objective optimization function
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix.
        -   addLS :   Bool   - Add line search (standard backtracking).
        
    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence
        - ittpSec : Float64 - Iterations per second of the method.
        -  Cflag  :  Bool   - The method converged (true or false)
    """
    #Start time
    start_time = time()
    
    #Init variables
    x_old = x0
    g_old = gradient(x_old) 
    gnorm = norm(g_old)
    k = 1
    archived_convergence_flag::Bool = false

    #Init the variables of this method
    sk, yk = fill!(similar(x0), 0), fill!(similar(x0), 0)
    
    #Init the optimization process
    while (gnorm >= tolerance && k < maxIters)
        #Computing and making the hessian 
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, g_old, epsilon)
 
        #Compute the coefficients
        V = [g_old, sk, yk]
        C = namgmSolver(h, g_old, V)

        #Update the sequence and the set of vectors
        v = -sum(V .* C)

        #Add the factor using line search
        fxk = fx(x_old)
        add_lineSearch ? alpha = backtrackWWC(fx, x_old, v, g_old, fxk) : alpha=1;
        x_new = x_old + alpha*v
        g_new = gradient(x_new)

        #Update the momentum and curvature variables
        sk = x_new - x_old
        yk = g_new - g_old

        #Reassign the variables
        g_old = g_new
        x_old = x_new

        #Update the gradient norm
        gnorm = norm(g_old)
        k+=1
        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("AMG")
            gnorm = Inf32
            k = maxIters
            break
        end
    end

    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)
    
    #Verification of convergence
    if (gnorm <= tolerance) && (k < maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution 
    displayResults("Oviedo", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x_old, k, ttime, gnorm, ittpSec, archived_convergence_flag  
end

function namgmGrads(fx, gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int, 
                    queue_size:: Int, hessian_mod, epsilon::Float64, add_lineSearch::Bool = false,
                    show_results::Bool = false)
    """NAMGM - Using the Gradient queue directive. 
    # Input:
        -    fx   : Function  - Objective optimization function.
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        - que_size:    Int   - Maximum number of elements in the queue (set of vectors).
        -hessian_mod: Call   - Modifier of the hessian matrix.
        -  addLS  :   Bool  - Add line search (standard backtracking)

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence
        - ittpSec : Float64 - Iterations per second of the method.
        -  Cflag  :  Bool   - The method converged (true or false)
    """
    #Start time
    start_time = time()

    #Init the variables.
    x = x0
    gk = gradient(x)
    gnorm = norm(gk)
    k = 1
    archived_convergence_flag::Bool = false

    #Init the variables of this method
    gradient_queue = Queue{Vector}()
    enqueue!(gradient_queue, gk)

    #Init the optimization process
    while (k < maxIters && gnorm >= tolerance)

        #Dequee if it's needed
        k > queue_size ? dequeue!(gradient_queue) :  nothing

        #Compute the hessian
        h = Matrix(Hessian(x))
        h = hessian_mod(h, gk, epsilon)
        
        #Collect the current elements in the queue to solve the optimization problem
        V = collect(gradient_queue)
        C = namgmSolver(h, gk, V)
        
        #Update the sequence and the set of vectors
        v = -sum(V .* C)

        #Add the coefficient using line search
        fxk = fx(x)
        add_lineSearch ? alpha = backtrackWWC(fx, x, v, gk, fxk) : alpha=1;
        x = x + alpha*v

        #Update the measured variables
        gk = gradient(x)
        gnorm = norm(gk) 
        enqueue!(gradient_queue, gk)
        k+=1
        
        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("Queue Gradients")
            gnorm = Inf32
            k = maxIters
            break
        end
    end
    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence
    if (gnorm <= tolerance) && (k < maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution 
    displayResults("Grads ", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x, k, ttime, gnorm, ittpSec, archived_convergence_flag  
end

function namgmRandomVectors(fx::Function, gradient, Hessian, x0:: Vector, tolerance:: Float64, maxIters:: Int,
                             randomSize:: Int, hessian_mod, epsilon::Float64, add_lineSearch::Bool = false, show_results::Bool = false)
    """NAMGM - Using the Random Vectors directive. 
    
    #Input
        -    fx   : Function  - Objective optimization function.
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        -randomSize:  Int.   - Number of random elements to take
        -hessian_mod: Call   - Modifier of the hessian matrix
        -  addLS  ;   Bool   - Add line search (standard backtracking)

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -   time  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence
        - ittpSec : Float64 - Iterations per second of the method.
        -  Cflag  :  Bool   - The method converged (true or false)
    """
    #Start time
    start_time = time()
    
    #Init the Variables
    x = x0
    gk = gradient(x)
    gnorm = norm(gk)
    k = 1
    archived_convergence_flag::Bool = false
    
    #Init the variables of this method
    dim = length(x0)

    #Optimization process
    while (k < maxIters && gnorm >= tolerance)
        #Initialize an empty list specifically for the random Vectors
        V = Vector{Vector{Float64}}()
        vectorsSample = randn(Float64, randomSize-1, dim)

        #Add gk as the FIRST whole vector
        push!(V, vec(Array(gk)))

        #Append the random vectors
        for i in 1:size(vectorsSample, 1)
            push!(V, vec(vectorsSample[i, :]))
        end

        #Get the approximation of the hessian
        h = Matrix(Hessian(x))
        h = hessian_mod(h, gk, epsilon)
  
        #Collect the current elements in the queue to solve the optimization problem
        C = namgmSolver(h, gk, V)

        #Update the sequence and the set of vectors
        v = -sum(V .* C)

        #Add the factor using line search
        fxk = fx(x)
        add_lineSearch ? alpha = backtrackWWC(fx, x, v, gk, fxk) : alpha=1;
        x = x + alpha*v

        #Update 
        gk = gradient(x)
        gnorm = norm(gk) 
        k+=1 

        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("Random")
            gnorm = Inf32
            k = maxIters
            break
        end
    end

    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence
    if (gnorm <= tolerance) && (k < maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution
    show_results ? displayResults("Random", k, ttime, gnorm, ittpSec, archived_convergence_flag) : nothing
    return x, k, ttime, gnorm, ittpSec, archived_convergence_flag 
end

function newtonMethod(fx::Function, gradient, hessian, x0::Vector, tolerance::Float64, maxIters:: Int, 
                        hessian_mod, epsilon::Float64, addLS::Bool = false)
    """
    Newton's method with applicable modifier in the Hessian matrix.
    (This is not the same as the BFGS method, such method use a unique modifier)
    # Input:
        -    fx   : Function  - Objective optimization function. 
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix
        -  addLS  :   Bool   - Add line search (standard backtracking)

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence    
        - ittpSec : Float64 - Iterations per second of the method.
        -  Cflag  :  Bool   - The method converged (true or false)
    """
    #Start time of the method
    start_time = time()
    
    #Init the variables
    x = x0
    gk = gradient(x)
    gnorm = norm(gk)
    k = 1
    archived_convergence_flag::Bool = false

    #Optimization process
    while (k < maxIters && gnorm >= tolerance)

        #Compute the approximation of the hessian matrix
        h = Matrix(hessian(x))
        h = hessian_mod(h, gk, epsilon)

        #Calculate the search direction and update the sequence point
        s = -h\gk
        
        #Add the factor using line search
        fxk = fx(x)
        addLS ? alpha = backtrackWWC(fx, x, s, gk, fxk) : alpha=1;
        x = x + alpha*s

        #Update 
        gk = gradient(x)
        gnorm = norm(gk) 
        k+=1

        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("Newton")
            gnorm = Inf32
            k = maxIters
            break
        end
    end

    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence
    if (gnorm <= tolerance) && (k < maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution
    displayResults("Newton", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x, k, ttime, gnorm, ittpSec, archived_convergence_flag  
end


function BFGSMethod(fx, gradient, hessian, x0::Vector, tolerance::Float64, maxIters:: Int, add_lineSearch::Bool = true)
    """
    Implementation of the BFGS method, inspired in the implementation of Nocedal and Stephen Wright.
    It uses line search (standard backtracking).
    # Input:
        -    fx   : Callable - Function
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -   addLS :   Bool   - Add line search (using backtracking, default: true)

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence    
        - ittpSec : Float64 - Iterations per second of the method.
        -  Cflag  :  Bool   - The method converged (true or false)

    # Remark:
        According to J. Nocedal & Wright the use of backtrack is discouraged or not situate for Quasi-Newton 
        methods. Nevertheless, to make comparative the results of the methods. 

    """

    #Start time of the method
    start_time = time()

    #Init the variables
    x_old = x0
    g_old = gradient(x_old)
    gnorm = norm(g_old)
    k = 1
    archived_convergence_flag::Bool = false
    
    #Init the other needed variables
    x_new = fill!(similar(x0), 0)
    g_new = fill!(similar(x0), 0)

    #Compute the inverse of the hessian (first approximation)
    hk = I
    try
        hk = inv(Symmetric(Matrix(hessian(x_old))))
    catch
        @warn "BFGS - Error trying to compute the inverse, using the identity as first approximation."
    end

    #Optimization process
    while (k < maxIters && gnorm >= tolerance)
        #Compute the direction
        pk = -hk*g_old

        #Update the point using weak wolfe condition
        actualfxk = fx(x_old)
        add_lineSearch ? alpha =backtrackWWC(fx, x_old, pk, g_old, actualfxk) : alpha = 1

        #Update the sequence
        x_new = x_old+alpha*pk
        g_new = gradient(x_new)

        #Update the Variables
        sk = x_new - x_old
        yk = g_new - g_old
        
        #Update the approximation of the hessian
        rho = 1/(dot(yk, sk) + 1e-9) #<- Added a small epsilon to avoid a possible NaN
        hk = (I-rho*sk*transpose(yk))*hk*(I-rho*yk*transpose(sk))+(rho*sk*transpose(sk))

        #Update the variables
        x_old = x_new
        g_old = g_new
        gnorm = norm(g_old) 
        k+=1

        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            println("Floating point overflow occurred, ending process.")
            gnorm = Inf32
            k = maxIters
            break
        end
    end

    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence
    if (gnorm <= tolerance) && (k <= maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution
    displayResults("BFGS", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x_old, k, ttime, gnorm, ittpSec, archived_convergence_flag 
end


function steepestMethod(fx::Function, gradient::Function, x0::Vector, tolerance::Float64, maxIters::Int, add_lineSearch::Bool = true, show_results::Bool = false,)
    """Gradient descent method that uses line-search strategy (backtrack).
    
    #Input:
        -    fx   : Callable - Function
        - Gradient: Callable - Gradient of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence
        -  add_LS :    Bool  - Add line search (standard backtracking) 
    #Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence    
        - ittpSec : Float64 - Iterations per second of the method.
        -  Cflag  :  Bool   - The method converged (true or false)
    
    """
    #Start time of the method
    start_time = time()

    #Init the variables
    x = x0
    gk = gradient(x)
    gnorm = norm(gk)
    k = 1 
    archived_convergence_flag::Bool = false

    #Optimization process
    while (k < maxIters && gnorm >= tolerance) 

        #Compute the right steep size
        actualfxk = fx(x)
        alpha = add_lineSearch ? backtrackWWC(fx, x, -gk, gk, actualfxk) : 1.0

        #Update the sequence
        x = x - alpha*gk
        gk = gradient(x)
        gnorm = norm(gk) 
        k+=1

        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("Step Descent")
            gnorm = Inf32
            k = maxIters
            break
        end
    end

    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence
    if (gnorm <= tolerance) && (k <= maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution
    displayResults("Gradient descent with LS", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x, k, ttime, gnorm, ittpSec, archived_convergence_flag  
end