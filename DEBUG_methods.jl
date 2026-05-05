"""

This is the same script that the NAMGM methods only that this file contains other measurements needed
for the analysis. For example, this methods measure the condition number of the NAMGM system, the used
paths of the methods for low dimensions (bidimensional), the gradient historial, and other standard 
measurements like solution, number of iterations, execution time and convergence flag.

"""

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
using Printf

#Include the utils file where is the Line Search method.
include("utils.jl")

function DEBUG_namgmSolver(Bk, gk, list_of_vectors, modifier::Function)
    """
    # Function description
    Function to solve the linear system of equations to obtain the corresponding 
    coefficients in the linear combination. It is supposed that the first vector
    in the array is the gradient vector ∇f(x).
    # Input
        - Bk: Matrix - Approximation of the hessian matrix.
        - gk: Vector - Gradient in the current iteration
        - lk: Array  - List of vectors to solve the system
        - modifier: Callable - Modifier of the System (to reduce the CN).
    # Output:
        - C:  Vector  - Constants of the linear combination of the vectors
        -CN: Float64  - Condition number of the system
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
            matrix[i, j] = V[i]' * V[j]
            matrix[j, i] = matrix[i, j]
        end
    end

    #Construct the response vector
    b = [gk' * V[i] for i in (1:n_rows)]

    #Here we need a try-catch implementation, we can use a small modification of the hessian matrix
    matrix = modifier(matrix)

    #Compute the condition number of the matrix
    val_max = eigmax(matrix)
    val_min = eigmin(matrix)
    cn = abs(val_max/val_min)

    #Try-Catch initialization variable
    C=nothing
    try
        C = matrix \ b
    catch
        C = namgmSolver(Bk, gk, list_of_vectors[1:end-1])
        C = [i <= length(C) ? C[i] : 0.0 for i in 1:length(list_of_vectors)]
    end
    return C, cn
end 


function DEBUG_namgmOviedo(fx::Function, gradient::Function, Hessian::Function, x0:: Vector,
                           tolerance:: Float64, maxIters:: Int, hessian_mod::Function,
                           epsilon::Float64, sys_mod::Function, addLS::Bool = false)
    """NAMGM - Using the Ovideo Directive. 
    # Input:
        -    fx   : Callable - Function 
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix.
        -   addLS :   Bool   - Add line search (standard backtracking, default: false)

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

    #Init the variables
    x_old = x0
    g_old = gradient(x_old) 
    gnorm = norm(g_old)
    k = 1
    archived_convergence_flag::Bool = false    

    #DEBUG variables
    gradient_his = []
    x_path = []
    condition_his = []
    dim = length(x0)
 
    #Steps matrix
    dim == 2 ? push!(x_path, x_old) : nothing

    #Creation of the others elements in the set of vectors
    sk, yk = zeros(Float64, dim), zeros(Float64, dim)

    #Init the optimization process
    while (gnorm >= tolerance && k < maxIters)
        #Computing and making the hessian 
        h = Matrix(Hessian(x_old))
        h = hessian_mod(h, g_old, epsilon)
 
        #Compute the coefficients
        V = [g_old, sk, yk]
        C, cn = DEBUG_namgmSolver(h, g_old, V, sys_mod)

        #Update the sequence and the set of vectors
        v = -sum(V .* C)

        #Add the coefficient using line search
        fxk = fx(x_old)
        addLS ? alpha = backtrackWWC(fx, x_old, v, g_old, fxk) : alpha=1;
        x_new = x_old + alpha*v
        g_new = gradient(x_new)

        #Update the momentum and curvature variables
        sk = x_new - x_old
        yk = g_new - g_old

        #Reassign the variables
        g_old = g_new
        x_old = x_new
        k+= 1

        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("Oviedo")
            gnorm = Inf32
            k = maxIters
            break
        end
        
        #Append the obtained values
        gnorm = norm(g_old) 
        push!(gradient_his, gnorm)
        push!(condition_his, cn)

        #If it's a bidimensional problem add the point
        dim == 2 ? push!(x_path, x_old) : nothing
    end
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence
    if (gnorm <= tolerance) && (k < maxIters)
        archived_convergence_flag = true 
    end

    #Print report of the execution 
    displayResults("Oviedo", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x_old, gradient_his, ttime, x_path, k, gnorm, condition_his, ittpSec, archived_convergence_flag 
end

function DEBUG_namgmGrads(fx::Function, gradient::Function, Hessian::Function, x0:: Vector, 
                          tolerance:: Float64, maxIters:: Int, queue_size:: Int, 
                          hessian_mod::Function, epsilon::Float64, sys_mod::Function, addLS::Bool = false)
    """NAMGM - Using the Gradient queue directive. 
    # Input:
        -    fx   : Callable - Objective function.
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        - que_size:    Int   - Maximum number of elements in the queue (set of vectors).
        -hessian_mod: Call   - Modifier of the hessian matrix.
        -   addLS :   Bool   - Add line search (standard backtracking, default: false)
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

    #DEBUG variables
    gradient_his = []
    x_path = []
    condition_his = []
    gradient_queue = Queue{Vector}()
    dim = length(x0)
    dim == 2 ? push!(x_path, x) : nothing

    #Init the variables of this method
    gradient_queue = Queue{Vector}()
    enqueue!(gradient_queue, gk)

    while (k < maxIters && gnorm >= tolerance)
        #Dequee if it's needed
        k > queue_size ? dequeue!(gradient_queue) :  nothing

        #Compute the hessian
        h = Matrix(Hessian(x))
        h = hessian_mod(h, gk, epsilon)
        
        #Collect the current elements in the queue to solve the optimization problem
        V = collect(gradient_queue)
        C, cn = DEBUG_namgmSolver(h, gk, V, sys_mod)

        #Update the squence and the set of vectors
        v = -sum(V .* C)

        #Add the coeficient using line search
        fxk = fx(x)
        addLS ? alpha = backtrackWWC(fx, x, v, gk, fxk) : alpha=1;
        x = x + alpha*v

        #Update 
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

        #Update the information
        push!(gradient_his, gnorm)
        push!(condition_his, cn)
        dim == 2 ? push!(x_path, x) : nothing
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
    displayResults("Grads", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x, gradient_his, ttime, x_path, k, gnorm, condition_his, ittpSec, archived_convergence_flag 
end


function DEBUG_namgmRandomVectors(fx::Function, gradient::Function, Hessian::Function, x0:: Vector, 
                                  tolerance:: Float64, maxIters:: Int, randomSize:: Int, 
                                  hessian_mod::Function, epsilon::Float64, sys_mod::Function, addLS::Bool = false)
    """NAMGM - Using the Random Vectors directive. 
    #Input
        -    fx   : Callable - Function.
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        -randomSize:  Int.   - Number of random elements to take
        -hessian_mod: Call   - Modifier of the hessian matrix
        -   addLS :   Bool   - Add line search (standard backtracking)

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

    #Init the Variables
    x = x0
    gk = gradient(x)
    gnorm = norm(gk)
    k = 1
    archived_convergence_flag::Bool = false

    #DEBUG Variables
    gradient_his = []
    condition_his = []
    x_path = []
    dim = length(x)

    #If the dimension of the objective function is 2 then we save the path
    dim == 2 ? push!(x_path, x) : nothing

    #Optimization process
    while (k < maxIters && gnorm >= tolerance)
        #Initialize an empty list specifically for Vectors
        V = Vector{Vector{Float64}}()
        vectorsSample = randn(Float64, randomSize-1, dim)

        #Add g_old as the FIRST whole vector
        push!(V, vec(Array(gk)))

        #We loop through rows and push them one by one
        for i in 1:size(vectorsSample, 1)
            push!(V, vec(vectorsSample[i, :]))
        end

        #Get the approximation of the hessian
        h = Matrix(Hessian(x))
        h = hessian_mod(h, gk, epsilon)
  
        #Collect the correct elements in the queue to solve the optimization problem
        C, cn = DEBUG_namgmSolver(h, gk, V, sys_mod)

        #Update the sequence and the set of vectors
        v = -sum(V .* C)
        
        #Add the coefficient using line search
        fxk = fx(x)
        addLS ? alpha = backtrackWWC(fx, x, v, gk, fxk) : alpha=1;
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

        #Update 
        push!(gradient_his, gnorm)
        push!(condition_his, cn)
        dim == 2 ? push!(x_path, x) : nothing
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
    displayResults("Random", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x, gradient_his, ttime, x_path, k, gnorm, condition_his, ittpSec, archived_convergence_flag 
end



function DEBUG_newtonMethod(fx::Function, gradient::Function, hessian::Function,  x0::Vector, 
                            tolerance::Float64, maxIters:: Int, hessian_mod, epsilon::Float64, addLS::Bool = false)
    """
    Newton's method with applicable modifier in the Hessian matrix
    (This is not the same as the BFGS method, such method use a unique modifier)
    # Input:
        -    fx   : Callable - Function.
        - Gradient: Callable - Gradient of the function.
        -  Hessian: Callable - Hessian of the function.
        -    x0   :  Vector  - Initial point of the sequence.
        -tolerance:   Float  - Minimum norm of the critical point
        - maxIters:    Int   - Maximum number of elements in the sequence.
        -hessian_mod: Call   - Modifier of the hessian matrix.
        -   addLS :   Bool   - Add line search (standard backtracking)

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence    
        - ittpSec : Float64 - Iterations per second of the method.
    """
    #Start time of the method
    start_time = time()

    #Init the variables
    x = x0
    gk = gradient(x)
    gnorm = norm(gk)
    k = 1
    archived_convergence_flag::Bool = false

    #DEBUG Variables
    gradient_his = []
    x_path = []
    dim = length(x0)
    dim == 2 ? push!(x_path, x) : nothing

    #Optimization process
    while (k < maxIters && gnorm >= tolerance)
        #Compute the approximation of the hessian matrix
        h = Matrix(hessian(x))
        h = hessian_mod(h, gk, epsilon)

        #Calculate the search direction and update the sequence point
        s = -h\gk

        #Add the coefficient using line search
        fxk = fx(x)
        addLS ? alpha = backtrackWWC(fx, x, s, gk, fxk) : alpha=1;
        
        #Update the variables
        x = x + alpha*s
        gk = gradient(x)
        gnorm = norm(gk) 
        k+=1 

        #Verification of no Inf or NaN value
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("Newton")
            gnorm = Inf32
            k = maxIters
            break
        end
        
        #If it's a bidimensional problem we add the path of the problem
        dim == 2 ? push!(x_path, x) : nothing
        push!(gradient_his, gnorm)
    
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
    return x, gradient_his, ttime, x_path, k, gnorm, ittpSec, archived_convergence_flag 
end


function DEBUG_BFGSMethod(fx::Function, gradient::Function, hessian::Function, x0::Vector, 
                        tolerance::Float64, maxIters:: Int, show_info::Bool = false, addLS::Bool = true)
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
        -   addLS :   Bool   - Add line search (standard backtracking, default: true)

    # Output:
        -    x    : Vector  - Final element of the sequence
        -    k    :  Int    - Number of iterations taken
        -  ttime  : Float64 - Execution time of the method in seconds
        -  gnorm  : Float64 - Gradient of the last element of the generated sequence    
        - ittpSec : Float64 - Iterations per second of the method.
    """
    #Start time of the method
    start_time = time()
    
    #Init the variables
    x_old = x0
    g_old = gradient(x_old)
    gnorm = norm(g_old)
    k = 1
    archived_convergence_flag::Bool = false
    
    #Debug variables
    gradient_his = []
    x_path = []
    dim = length(x0)
    dim == 2 ? push!(x_path, x_old) : nothing

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
   
    while (k < maxIters && gnorm >= tolerance)
        #Compute the direction
        pk = -hk*g_old

        #Update the point using weak wolfe condition
        actualfxk = fx(x_old)
        addLS ? alpha = backtrackWWC(fx, x_old, pk, g_old, actualfxk, show_info = show_info) : alpha = 1;
        x_new = x_old+alpha*pk
        g_new = gradient(x_new)

        #Update the Variables
        sk = x_new - x_old
        yk = g_new - g_old
        
        #Update the approximation of the hessian
        rho = 1/(dot(yk, sk) + 1e-9)
        hk = (I-rho*sk*transpose(yk))*hk*(I-rho*yk*transpose(sk))+(rho*sk*transpose(sk))

        #Update the variables
        x_old = x_new
        g_old = g_new
        gnorm = norm(g_old) 
        k+=1

        #Verification of not divergence
        if isinf(gnorm) || isnan(gnorm) 
            displayOverflowError("BFGS")
            gnorm = Inf32
            k = maxIters
            break
        end

        #Push the new results in the historial
        push!(gradient_his, gnorm)
        dim == 2 ? push!(x_path, x_old) : nothing
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
    displayResults("BFGS", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x_old, gradient_his, ttime, x_path, k, gnorm, ittpSec, archived_convergence_flag 
end

function DEGUB_steepestMethod(fx::Function, gradient::Function, x0::Vector, tolerance::Float64, maxIters::Int, add_lineSearch::Bool = true)
    """Gradient descent method that uses line-search strategy.
    
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
    
    #DEBUG variables
    gradient_his = []
    x_path = []
    dim = length(x0)
    dim == 2 ? push!(x_path, x) : nothing

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

        #Push the new results in the historial
        push!(gradient_his, gnorm)
        dim == 2 ? push!(x_path, x) : nothing
    end

    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Verification of convergence    
    archived_convergence_flag = (gnorm <= tolerance) && (k <= maxIters) ? true : false

    #Print report of the execution
    displayResults("Gradient descent with LS", k, ttime, gnorm, ittpSec, archived_convergence_flag)
    return x, gradient_his, ttime, x_path, k, gnorm, ittpSec, archived_convergence_flag  
end