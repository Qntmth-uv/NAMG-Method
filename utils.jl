using DataFrames

using CSV



function getIterationSpeed(numberOfIters, time::Float64)
    """Compute the avrg iteration speed."""
    return numberOfIters/time
end


function createCSV(arrayofHistorials, CSV_fileName::String, headers )
    """Function that write a CSV given the historial of the executions"""
    #Find the length of the longest vector
    max_len = maximum(length, arrayofHistorials)

    #Create new padded vectors
    #This creates a new vector for each column, filling the gap with `missing`
    arrayofHistorials = [ [v; fill(missing, max_len - length(v))] for v in arrayofHistorials]

    #Save the CSV fike
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

function getHeatmapCosine(listOfVectors)
    """Creation """
    dim = length(listOfVectors)
    similaritys = zeros(Float64, dim, dim)

    for i in 1:dim
        for j in i:dim
            similaritys[i, j] = fastCosineSim(listOfVectors[i], listOfVectors[j])
            similaritys[j, i] = similaritys[i, j]
        end
    end
    return similaritys
end


function getConditionNumber(M)
    """Function to get the Condition Number of the NAMGM_system, this is posible because
    the dimension of this matrix is easy to compute due his dimension.
    
    # Input:
        - M: Matrix - Hψ(B_k, g_k, V) matrix of the linear system
    
    # Output:
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

# Definir una función auxiliar para el mapeo
function get_modifier(mod_name)
    if mod_name == "eigen"
        return modifyHessian_Eigen
    elseif mod_name == "diag"
        return diagonalModifier_Hessian
    elseif mod_name == "tridiag"
        return tridiagonalModifier_Hessian
    elseif mod_name == "remove"
        return removeConvergenceModifier
    else
        return notModifierHessian
    end
end


using LinearAlgebra

"""
    backtracking_line_search(f, x, p, ∇f_x, f_x; α=1.0, ρ=0.5, c1=1e-4)

Performs a backtracking line search to satisfy the Armijo (Weak Wolfe) condition.
Based on Algorithm 3.1 from Nocedal & Wright.

# Arguments
- `f`: The objective function f(x).
- `x`: Current position vector.
- `p`: Search direction vector.
- `∇f_x`: The gradient at current x (pre-computed).
- `f_x`: The function value at current x (pre-computed).

# Keywords
- `α`: Initial step size (default 1.0, typical for Newton methods).
- `ρ`: Contraction factor (default 0.5).
- `c1`: Armijo parameter (default 1e-4).

# Returns
- `α`: The accepted step size.
- `f_new`: The function value at the new point (to avoid re-evaluating later).
"""
function backtrackWWC(f::Function, x::AbstractVector, p::AbstractVector, ∇f_x::AbstractVector, 
                      f_x::Real; α::Real=1.0, ρ::Real=0.5, c1::Real=1e-4)
    
    # 1. Pre-compute the directional derivative (slope)
    # This is the 'm' in f(x + αp) ≤ f(x) + c1 * α * m
    slope = dot(∇f_x, p)

    # Sanity check: Ensure p is a descent direction

    if slope > 0
        @warn "Search direction p is not a descent direction (slope > 0). Line search may fail."
    end

    # 2. Backtracking Loop
    # We limit the loop to avoid infinite cycles in case of numerical errors
    max_iter = 100
    iter = 0
    
    # Evaluate target function once inside the loop structure
    x_new = x + α * p
    f_new = f(x_new)

    # While the Armijo condition is NOT met:
    # f(x + αp) > f(x) + c1 * α * (∇f'p)
    while f_new > f_x + c1 * α * slope && iter < max_iter
        # Backtrack: reduce alpha by contraction factor ρ
        α *= ρ
        
        # Update candidate point
        x_new = x + α * p
        f_new = f(x_new)
        
        iter += 1
    end

    if iter == max_iter
        @warn "Line search failed to converge within max iterations. Returning small α."
    end

    return α
end