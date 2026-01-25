"""
This is the same script that the NAMGM methods
only that this file contains other measurements needed
for the analysis. For example, this methods measure 
the condition number of the NAMGM system, the used
paths of the methods for low dimensions (bidimensional),
the gradient historial, and other standard measurements
like solution, number of iterations, executation time
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

include("utils.jl")


#Need to implement a BFGS method to compare.
function DEBUG_namgmSolver(Bk, gk, list_of_vectors, modifier::Function)
    """Suponemos que el primer vector en la lista es necesariamente el vector del gradiente
    
    # Input
        - Bk: Matrix - Approximation of the hessian matrix.
        - gk: Vector - Gradient in the current iteration
        - lk: Array  - List of vectors to solve the system
        - modifier: Callable - Modifier of the System (to reduce the CN).
    # Output:
        - C:  Vector  - Constans of the linear combination of the vectors
        -CN: Float64  - Condition number of the system
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
    matrix = modifier(matrix)

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
                           epsilon::Float64, sys_mod::Function)
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
        C, cn = DEBUG_namgmSolver(h, g_old, V, sys_mod)

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
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Print report of the execution 
    @printf("Execution Info - Oviedo| Iters: %6d | TTime: %.5f | LastNorm: %1.5e | Iterations/sec: %-6.5f |\n", k, ttime, gnorm, ittpSec)
    return x_old, gradient_his, ttime, x_path, k, gnorm, condition_his, ittpSec 
end

function DEBUG_namgmGrads(gradient::Function, Hessian::Function, x0:: Vector, 
                          tolerance:: Float64, maxIters:: Int, queue_size:: Int, 
                          hessian_mod::Function, epsilon::Float64, sys_mod::Function)
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
        C, cn = DEBUG_namgmSolver(h, g_old, V, sys_mod)

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
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Print report of the execution 
    @printf("Execution Info - Grads | Iters: %6d | TTime: %.5f | LastNorm: %1.5e | Iterations/sec: %-6.5f |\n", k, ttime, gnorm, ittpSec)
    return x_old, gradient_his, ttime, x_path, k, gnorm, condition_his, ittpSec 
end


function DEBUG_namgmRandomVectors(gradient::Function, Hessian::Function, x0:: Vector, 
                                  tolerance:: Float64, maxIters:: Int, randomSize:: Int, 
                                  hessian_mod::Function, epsilon::Float64, sys_mod::Function)
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
        vectorsSample = randn(Float64, randomSize-1, dim)

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
        C, cn = DEBUG_namgmSolver(h, g_old, V, sys_mod)

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
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Print report of the execution 
    @printf("Execution Info - Random| Iters: %6d | TTime: %.5f | LastNorm: %1.5e | Iterations/sec: %-6.5f |\n", k, ttime, gnorm, ittpSec)
    return x_old, gradient_his, ttime, x_path, k, gnorm, condition_his, ittpSec 
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
    k = 0


    if dim == 2
        push!(x_path, x0)
    end


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
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Print report of the execution  
    @printf("Execution Info - Newton| Iters: %6d | TTime: %.5f | LastNorm: %1.5e | Iterations/sec: %-6.5f |\n", k, ttime, gnorm, ittpSec)
    return x, gradient_his, ttime, x_path, k, gnorm, ittpSec 
end


function DEBUG_BFGSMethod(fx::Function, gradient::Function, hessian::Function, x0::Vector, tolerance::Float64, maxIters:: Int)
    #Start time of the method
    start_time = time()
    dim = length(x0)

    #Debug variables
    gradient_his = []
    x_path = []
    dim = length(x0)
    if dim == 2
        push!(x_path, x0)
    end
    k = 0

    #Init the variables
    x_old = x0
    x_new = zeros(Float64, length(dim)) 
    g_new = zeros(Float64, length(dim))

    #Compute the inverse of the hessian (first approximation)
    hk = I
    try
        hk = inv(Symmetric(Matrix(hessian(x_old))))
    catch
        @warn "BFGS - Error trying to compute the inverse, using the identity as first approximation."
    end
   
    gnorm = Inf
    while (k < maxIters && gnorm >= tolerance)
        #Compute the gradient of the matrix
        g_old = gradient(x_old)

        #Compute the direction
        pk = -hk*g_old

        #Update the point using weak wolfe condition
        actualfxk = fx(x_old)
        alpha = backtrackWWC(fx, x_old, pk, g_old, actualfxk)
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

        #Push the new results in the historial
        push!(gradient_his, gnorm)
        if dim == 2
            push!(x_path, x_old)
        end
        k+=1
    end
    #Finalization of the method
    end_time = time()
    ttime = end_time-start_time
    ittpSec = getIterationSpeed(k, ttime)

    #Print report of the execution 
    @printf("Execution Info - BFGSM | Iters: %6d | TTime: %.5f | LastNorm: %1.5e | Iterations/sec: %-6.5f |\n", k, ttime, gnorm, ittpSec)
    return x_old, gradient_his, ttime, x_path, k, gnorm, ittpSec 
end