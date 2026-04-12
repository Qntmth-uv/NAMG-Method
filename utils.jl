using DataFrames
using LinearAlgebra
using CSV
using Printf

include("hessian_mod.jl")

function getIterationSpeed(numberOfIters, time::Float64)
    """Compute the average iteration speed."""
    return numberOfIters/time
end



function createCSV(arrayofHistorials, CSV_fileName::String, headers )
    """Function that write a CSV given the historial of the executions"""
    #Find the length of the longest vector
    max_len = maximum(length, arrayofHistorials)

    #Create new padded vectors
    #This creates a new vector for each column, filling the gap with `missing`
    arrayofHistorials = [ [v; fill(missing, max_len - length(v))] for v in arrayofHistorials]

    #Save the CSV file
    df = DataFrame(arrayofHistorials, headers)

    #Write the CSV file
    CSV.write(CSV_fileName, df)
end

function fastCosineSim(u::Vector, v::Vector)
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

function getConditionNumber(M)
    """Function to get the Condition Number of the NAMGM_system, this is possible because
    the dimension of this matrix is easy to compute due his dimension.
    
    # Input:
        - M: Matrix - Hψ(B_k, g_k, V) matrix of the linear system
    
    # Outputs:
        - C: Float64 - Condition number of Hψ(B_k, g_k, V) 
        - em:Float64 - Smallest eigenvalue of M
        - eM:Float64 - Biggest eigenvalue of M
    """
    #Computation of the eigenvalues and the condition number
    M = Symmetric(M)
    val_max = eigmax(M)
    val_min = eigmin(M)
    return abs(val_max/val_min), val_min, val_max
end

function elementsTestFunction(name::String)
    """Function to use the function, gradient, Hessian of a function, also
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


function get_modifier(mod_name)
    """Function to obtain the given modifier of the hessian"""
    if mod_name == "eigen"
        return modifyHessian_Eigen
    elseif mod_name == "diag"
        return diagonalModifier_Hessian
    elseif mod_name == "sabsdiag"
        return diagonalModifier_sabs_Hessian
    elseif  mod_name == "maxdiag"
        return diagonalModifier_max_Hessian
    elseif mod_name == "tridiag"
        return tridiagonalModifier_Hessian
    elseif mod_name == "remove"
        return removeConvergenceModifier
    else
        return notModifierHessian
    end
end

function backtrackWWC(f::Function, x::AbstractVector, p::AbstractVector, g_x::AbstractVector, 
                      f_x::Real; alpha::Real=1.0, factor::Real=0.5, c1::Real=1e-4, show_info::Bool = false)
    """Performs a backtracking line search to satisfy the Armijo (Weak Wolfe) condition.
    Based on Algorithm 3.1 from Nocedal & Wright.

    # Inputs
        - f: The objective function f(x).
        - x: Current position vector.
        - p: Search direction vector.
        - g_x: The gradient at current x (pre-computed).
        - f_x: The function value at current x (pre-computed).

    # Returns
        - alpha: The accepted step size.

    # Keywords
        - alpha: Initial step size (default 1.0, typical for Newton methods).
        - factor: Contraction factor (default 0.5).
        - c1: Armijo parameter (default 1e-4).

    """   
    #Limits of the backtracking variables
    max_iter = 100
    iter = 0
    
    #This is the 'm' in f(x + αp) ≤ f(x) + c1 * α * m
    slope = dot(g_x, p)

    #Sanity check: Ensure p is a descent direction
    slope > 0 && show_info ? println("Search direction p is not a descent direction (slope > 0). Line search may fail.") : nothing

    #Evaluate target function once inside the loop structure
    x_new = x + alpha * p
    f_new = f(x_new)

    #While the Armijo condition is satisfied:
    while f_new > f_x + c1 * alpha * slope && iter < max_iter
        #Contract  alpha by factor ρ
        alpha *= factor
        
        # Update candidate point
        x_new = x + alpha * p
        f_new = f(x_new)
        iter += 1
    end

    #Maximum number of iterations reached 
    iter == maximum && show_info ? println("Line search failed to converge within max iterations. Returning small step size.") : nothing
    return alpha
end

function displayResults(name::String, iters::Any, Ttime:: Any, Gnorm::Any, IttpS::Any, convergence::Bool)
    @printf("Execution Info - %s | Iters: %6d | TTime: %.5f | LastNorm: %1.5e | Iterations/sec: %-6.5f | Convergence: %s |\n", 
            name, iters, Ttime, Gnorm, IttpS, convergence)
end

function displayOverflowError(method_name::String)
    """Function to display a message of overflow error over one method.
    
    #Input:
        - method_name: String 

    #Output:
        - VOID
    """
    println("$method_name - Floating point overflow occurred, ending process.")
end