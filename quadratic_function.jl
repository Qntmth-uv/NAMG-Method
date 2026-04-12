using LinearAlgebra
using BenchmarkTools
include("./NAMGM_methods.jl")
include("./hessian_mod.jl")
include("utils.jl")

"""
Execution Examples
"""


#Constant Variable/s
const minValue = 2.220446049250313e-16
const USE_LS = false

#Structure of the data (it only save the parameters of the Quadratic functions)
struct GeneralQuadraticFunction
    A::Matrix{Float64}
    b::Vector{Float64}
end

#Generator of objective function (quadratic function)
function giveFunction_Qx_Q(f::GeneralQuadraticFunction)
    """Anonymous Function evaluator generator.
    #Input:
        - f: GeneralQuadraticFunction - Data from the Quadratic problem
    #Output:
        - f(x): Anonymous function to evaluate the function using the given data
    """
    return x -> (0.5 * x' * f.A * x) + (f.b' * x)
end

function giveGradient_Qx(f::GeneralQuadraticFunction)
    """Gradient generator Anonymous Function.
    #Input:
        - f: GeneralQuadraticFunction - Data from the Quadratic problem
    #Output:
        - g(x): Anonymous function to evaluate the gradient using the given data
    """
    return x -> f.A * x + f.b
end

function giveHessian_Qx(f::GeneralQuadraticFunction)
    """Anonymous Hessian evaluator generator.
    #Input:
        - f: GeneralQuadraticFunction - Data from the Quadratic problem
    #Output:
        - H(x): Anonymous function to evaluate the Hessian using the given data
    """
    return x -> f.A
end

function QuadraticFunctions(A::Matrix, b::Vector)
    """Initializer of a Quadratic function and its generators.
    #Input:
        - A: Matrix - Matrix of the Quadratic problem
        - b: Vector - Vector of the Quadratic problem
    #Output:
        - f(x): Anonymous function to evaluate the function using the given data
        - g(x): Anonymous function to evaluate the gradient using the given data
        - h(x): Anonymous function to evaluate the hessian using the given data
    
    """
    #Initialize the structure
    quadraticStructure = GeneralQuadraticFunction(A, b)

    #Creation of the Anonymous functions
    f = giveFunction_Qx_Q(quadraticStructure)
    g = giveGradient_Qx(quadraticStructure)
    h = giveHessian_Qx(quadraticStructure)
    return f, g, h
end

function generatorValues_Dense3(dim::Int, exponentInteger::Int)
    """Creation of the matrix of the Quadratic function using Ovideo's procedure for
        dense matrices of type 3.
    
    # Inputs:
        - dim: Int - Dimension of the created matrix
        - ek:  Int - Integer put on the exponent of the condition number of the matrix 

    # Outputs:
        - A: Matrix - Specific matrix with a fix condition number

    # Remarks:
        - According to BenchmarkTools the creation of a matrix with dimension 10000
        uses 43.895s and (36 allocations: 5.22 GiB) on a Macbook-Air M4 16GB RAM.
        Therefore the creation of matrices with a higher dimension can be a hard task
        for this procedure. The most difficult part of this algorithm is the QR factorization.
    
    """
    #Creation of a random matrix and the extraction of the orthogonal matrix
    Q, _ = qr(rand(Float64, dim, dim))

    #Fill the matrix
    D = Diagonal(Float64[ exp(((i-1)/dim)*exponentInteger) for i in 1:dim ])

    #Construct the Matrix
    A = Q * D * Q'
    
    #Construct the vector
    b = Q*rand(Float64, dim)
    return A, b
end

function main()
    #Parameters of the Strongly convex function problem
    dim = 1000
    nIters = 1000
    tol = 1e-8 
    lqueue = 50

    #Creation of the variables
    A, b = generatorValues_Dense3(dim, 10)
    f, g, h = QuadraticFunctions(A, b)
    
    # Initialize random vector of same dimension
    x0 = rand(dim)

    #Modifier to be used
    mod = notModifierHessian

    #Variables in a TRY-CATCH
    xf_newton, historial_newton, t_newton, xP_newton, iterNewton, normNewton, ttpSN, flagNewton = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing

    #Run the NAMGM algorithms
    xf_SGD, iterSGD, t_sgd, normSGD, ttpSGD, flagSGD = steepestMethod(f, g, x0, tol, nIters)
    xf_Oviedo, iterOviedo, t_oviedo, normOviedo, ttpSO, flagOviedo = namgmOviedo(f, g, h, x0, tol, nIters, mod, minValue, USE_LS)
    xf_Queue, iterQueue, t_queue, normQueue, ttpSQ, flagQueue = namgmGrads(f, g, h, x0, tol, nIters, lqueue, mod, minValue, USE_LS)
    xf_Random, iterRandom, t_random, normRandom, ttpSR, flagRandom = namgmRandomVectors(f, g, h, x0, tol, nIters, lqueue-1, mod, minValue, true, USE_LS)
    xf_bfgs, iterBFGS, t_bfgs, normBFGS, ttpSB, flagBFGS = BFGSMethod(f, g, h, x0, tol, nIters)
    try
        xf_newton, iterNewton, t_newton, normNewton, ttpSN, flagNewton = newtonMethod(f, g, h, x0, tol, nIters, mod, minValue, USE_LS)
    catch e
        println("An error was found in the execution of the Newton method: $e")
        xf_newton, iterNewton, t_newton, normNewton, ttpSN, flagNewton = x0, Inf, Inf, Inf, 0.0, false
    end    

    #Creation of Dataframe for the results
    times = [t_sgd, t_oviedo, t_queue, t_random, t_bfgs, t_newton]
    lastGrad = [normSGD, normOviedo, normQueue, normRandom, normBFGS, normNewton]
    iterations =[iterSGD ,iterOviedo, iterQueue, iterRandom, iterBFGS,iterNewton]
    data = [iterations, lastGrad, times]
    
    #Header of the DF
    headers = ["iterations", "Last Gradient", "Execution time"]

    #Save the CSV file
    df = DataFrame(data, headers)

    #Write the CSV file
    #CSV.write("csvs/quadratic_results.csv", df)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end