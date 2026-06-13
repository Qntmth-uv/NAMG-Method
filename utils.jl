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

function elementsTestFunction(name::String, dimension_var::Int = -1, variable_name::String = "N")
    """
    # Definition.

    Function to use the function, gradient, Hessian of a function, also
    return the x_minima and the nlp_problem to close it latter.
    
    ## Input.

    - ``name``: String - Name of the optimization problem (SIF name problem)
    -``dimension_var``: Int - Variable dimension from the optimization problem (default -1)
    
    The ``dimension_var == -1`` refers to the default value of the optimization problem variable.
    Most of the problems does not have a several dimension definitions, nevertheless, functions used
    in the Robust Experiments have such feature.

    ## Output.

    - (objective, gradient, hessian, X0, NLP Problem)
    
    Where objective, gradient, hessian, are the analytical definition of the function. The point
    X0 is the given initial point of the problem, and NLP is the environment the non linear problem.
    """

    #Create a model for the name of the problem depending the dimension of the problem
    nlp_problem = (dimension_var == -1) ? CUTEstModel(name) : CUTEstModel(name, "-param", "$variable_name=$dimension_var") 

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

    println("Number of variables: ", nlp_problem.meta.nvar)
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
using LinearAlgebra

function LS_cubic_interpolation(fx::Function, gx::Function, x::AbstractVector, p::AbstractVector, efficient_mode::Bool = false; initial_alpha::Real = 1.0, c1::Real = 1e-4)
    #We Compute baseline values at alpha = 0
    phi_0 = fx(x)
    g0 = gx(x)
    phi_prime_0 = dot(g0, p) # This is phi'(0), a scalar

    #Verification that the given direction is a descent direction
    if phi_prime_0 >= 0
        println("Search direction p is not a descent direction (slope > 0). Line search may fail.") : nothing
    end

    #First iteration: Try initial_alpha
    alpha_0 = initial_alpha
    phi_alpha_0 = fx(x + alpha_0 * p)

    #If it satisfies the Armijo condition immediately, we return it
    if phi_alpha_0 <= phi_0 + c1 * alpha_0 * phi_prime_0
        return alpha_0
    end

    #If its not First we use the quadratic interpolation using phi(0), phi'(0), and phi(alpha_0)
    alpha_1 = - (phi_prime_0 * alpha_0^2) / (2 * (phi_alpha_0 - phi_0 - phi_prime_0 * alpha_0))
    
    #Safeguard if alpha_1 so it doesn't shrink too fast or too slow
    alpha_1 = max(0.1 * alpha_0, min(0.9 * alpha_0, alpha_1))
    phi_alpha_1 = fx(x + alpha_1 * p)

    #Again if this quadratic approximation satisfies the Armijo condition, we return it
    if phi_alpha_1 <= phi_0 + c1 * alpha_1 * phi_prime_0
        return alpha_1
    end

    #For now on we use the cubic interpolation
    alpha_prev = alpha_0
    phi_prev = phi_alpha_0

    alpha_curr = alpha_1
    phi_curr = phi_alpha_1
    
    #Uses only function evaluations from current and previous alpha elements
    if efficient_mode

        while phi_curr > phi_0 + c1 * alpha_curr * phi_prime_0
            det = (alpha_curr^2 * alpha_prev^2) * (alpha_curr - alpha_prev)
            
            #Analytical inversion of Nocedal eq. (3.57)
            a = (alpha_prev^2 * (phi_curr - phi_0 - alpha_curr * phi_prime_0) - alpha_curr^2 * (phi_prev - phi_0 - alpha_prev * phi_prime_0)) / det
            b = (-alpha_prev^3 * (phi_curr - phi_0 - alpha_curr * phi_prime_0) + alpha_curr^3 * (phi_prev - phi_0 - alpha_prev * phi_prime_0)) / det
            
            #We handle the posivility of having a negative discriminan, in other words, complex solutions.
            discriminant = b^2 - 3 * a * phi_prime_0
            if discriminant < 0
                alpha_new = alpha_curr / 2 # Fallback to bisection if roots are complex
            else
                alpha_new = (-b + sqrt(discriminant)) / (3 * a)
            end

            #Apply safeguards
            alpha_new = max(0.1 * alpha_curr, min(0.9 * alpha_curr, alpha_new))

            #Update tracking variables
            alpha_prev = alpha_curr
            phi_prev = phi_curr
            alpha_curr = alpha_new
            phi_curr = fx(x + alpha_curr * p)
        end
    else
        #Standard mode: Uses both function values and gradients (e.g., Wolfe conditions Zoom)
        phi_prime_prev = dot(gx(x + alpha_prev * p), p)
        phi_prime_curr = dot(gx(x + alpha_curr * p), p)

        while phi_curr > phi_0 + c1 * alpha_curr * phi_prime_0
            d1 = phi_prime_prev + phi_prime_curr - 3 * (phi_prev - phi_curr) / (alpha_prev - alpha_curr)
            discriminant = d1^2 - phi_prime_prev * phi_prime_curr
            
            #Again we handle the posivility of having complex roots
            if discriminant < 0
                alpha_new = alpha_curr / 2
            else
                d2 = sign(alpha_curr - alpha_prev) * sqrt(discriminant)
                alpha_new = alpha_curr - (alpha_curr - alpha_prev) * ((phi_prime_curr + d2 - d1) / (phi_prime_curr - phi_prime_prev + 2 * d2))
            end

            #Apply safeguards
            alpha_new = max(0.1 * alpha_curr, min(0.9 * alpha_curr, alpha_new))

            #Update tracking variables
            alpha_prev = alpha_curr
            phi_prev = phi_curr
            phi_prime_prev = phi_prime_curr

            alpha_curr = alpha_new
            phi_curr = fx(x + alpha_curr * p)
            phi_prime_curr = dot(gx(x + alpha_curr * p), p)
        end
    end

    return alpha_curr
end

function line_search_SWC(fx::Function, gx::Function, x::AbstractVector, p::AbstractVector;
    alpha_1::Real = 1.0, alpha_max::Real = 5.0, c1::Real = 1e-4, c2::Real = 0.9, max_iter::Int = 10)
    
    #Definition of ϕ(.) functions.
    phi(alpha) = fx(x + alpha * p)
    phi_prime(alpha) = dot(gx(x + alpha * p), p)

    #Constant variables
    phi_0 = phi(0.0)
    phi_prime_0 = phi_prime(0.0)

    #Assertion that the given direction is a decreasing direction.
    if phi_prime_0 >= 0
        println("The search direction p is not a descent direction!")
    end

    #Declaration of previous variables
    alpha_prev = 0.0
    phi_prev = phi_0
    alpha_curr = alpha_1

    #Search loop
    for i in 1:max_iter
        phi_curr = phi(alpha_curr)

        # Condition 1: Check if current step violates sufficient decrease 
        # or if the function value increased compared to the previous step
        if phi_curr > phi_0 + c1 * alpha_curr * phi_prime_0 || (i > 1 && phi_curr >= phi_prev)
            return zoom(phi, phi_prime, alpha_prev, alpha_curr, phi_0, phi_prime_0, c1, c2)
        end

        phi_prime_curr = phi_prime(alpha_curr)

        # Condition 2: Check Strong Wolfe curvature condition
        if abs(phi_prime_curr) <= -c2 * phi_prime_0
            return alpha_curr
        end

        # Condition 3: If the derivative is positive, we passed a local minimum
        if phi_prime_curr >= 0
            return zoom(phi, phi_prime, alpha_curr, alpha_prev, phi_0, phi_prime_0, c1, c2)
        end

        # Step bound update (e.g., double the step size or choose a point in between)
        alpha_prev = alpha_curr
        phi_prev = phi_curr
        alpha_curr = min(2 * alpha_curr, alpha_max)

        # Guard against reaching max bounds
        if alpha_curr == alpha_max
            return alpha_max
        end
    end

    return alpha_curr
end

"""
    zoom(phi, phi_prime, alpha_lo, alpha_hi, phi_0, phi_prime_0, c1, c2; max_iter=10)

Algorithm 3.6 (Zoom) from Nocedal & Wright. Interpolates between alpha_lo and alpha_hi
to pinpoint a valid step length satisfying the Strong Wolfe Conditions.
"""
function zoom(phi::Function, phi_prime::Function, alpha_lo::Real, alpha_hi::Real, phi_0::Real, phi_prime_0::Real, c1::Real, c2::Real;max_iter::Int = 10)
    for j in 1:max_iter
        # Evaluate current endpoints
        phi_lo = phi(alpha_lo)
        phi_prime_lo = phi_prime(alpha_lo)
        
        phi_hi = phi(alpha_hi)
        phi_prime_hi = phi_prime(alpha_hi)

        #--- Cubic Interpolation Step ---
        d1 = phi_prime_lo + phi_prime_hi - 3 * (phi_lo - phi_hi) / (alpha_lo - alpha_hi)
        discriminant = d1^2 - phi_prime_lo * phi_prime_hi
        
        alpha_j = alpha_lo # Fallback initialization

        if discriminant >= 0
            d2 = sign(alpha_hi - alpha_lo) * sqrt(discriminant)
            alpha_j = alpha_hi - (alpha_hi - alpha_lo) * ((phi_prime_hi + d2 - d1) / (phi_prime_hi - phi_prime_lo + 2 * d2))
        end

        # Safeguard: If interpolation fails or falls out of bounds, use bisection
        alpha_min = min(alpha_lo, alpha_hi)
        alpha_max = max(alpha_lo, alpha_hi)
        # We also enforce that alpha_j isn't dangerously close to the boundaries
        if discriminant < 0 || alpha_j <= alpha_min + 1e-10 || alpha_j >= alpha_max - 1e-10
            alpha_j = alpha_lo + 0.5 * (alpha_hi - alpha_lo)
        end

        #--- Update Interval ---
        phi_j = phi(alpha_j)

        if phi_j > phi_0 + c1 * alpha_j * phi_prime_0 || phi_j >= phi_lo
            alpha_hi = alpha_j
        else
            phi_prime_j = phi_prime(alpha_j)
            
            # Success! Found a point satisfying Strong Wolfe
            if abs(phi_prime_j) <= -c2 * phi_prime_0
                return alpha_j
            end
            
            # If the slope forces us away from alpha_hi, replace alpha_hi with alpha_lo
            if phi_prime_j * (alpha_hi - alpha_lo) >= 0
                alpha_hi = alpha_lo
            end
            
            alpha_lo = alpha_j
        end
    end

    return alpha_lo
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