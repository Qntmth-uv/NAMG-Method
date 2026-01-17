using LinearAlgebra
using BenchmarkTools
include("./NAMGM_method.jl")
include("./hessian_mod.jl")

# 1. La estructura de datos (Solo guarda los parámetros fijos)
struct GeneralQuadraticFunction
    A::Matrix{Float64}
    b::Vector{Float64}
end

# --- GENERADORES DE FUNCIONES (CLOSURES) ---
function giveFunction_Qx_Q(f::GeneralQuadraticFunction)
    """Anonim Function evaluator generator.
    #Input:
        - f: GeneralQuadraticFunction - Data from the Quadratic problem
    #Output:
        - f(x): Anonim function to evaluate the function using the given data
    """
    return x -> (0.5 * x' * f.A * x) + (f.b' * x)
end


function giveGradient_Qx(f::GeneralQuadraticFunction)
    """Gradient generator anonim fucntion.
    #Input:
        - f: GeneralQuadraticFunction - Data from the Quadratic problem
    #Output:
        - g(x): Anonim function to evaluate the gradient using the given data
    """
    return x -> f.A * x + f.b
end


function giveHessian_Qx(f::GeneralQuadraticFunction)
    """Anonim Hessian evaluator generator.
    #Input:
        - f: GeneralQuadraticFunction - Data from the Quadratic problem
    #Output:
        - f(x): Anonim function to evaluate the Hessian using the given data
    """
    return x -> f.A
end


function QuadraticFunctions(A::Matrix, b::Vector)
    """Initilizer of a Quadratic function and its generators.
    #Input:
        - A: Matrix - Matrix of the Quadratic problem
        - b: Vector - Vector of the Quadratic problem
    #Output:
        - f(x): Anonim function to evaluate the function using the given data
        - g(x): Anonim function to evaluate the gradient using the given data
        - h(x): Anonim function to evaluate the hessian using the given data
    
    """
    #Initialize the structure
    quadraticStructure = GeneralQuadraticFunction(A, b)

    #Creation of the anonim functions
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
        - According to BenchmarkTools the creation of a matrix with dimention 10000
        uses 43.895s and (36 allocations: 5.22 GiB) on a Macbook Air M4 16GB URAM.
        The refore the creation of matrices with a higher dimension can be a hard task
        for this procedure. The most dificult part of this algoritm is the QR factorization.
    
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
    # --- Configuración de datos ---
    # Creamos una matriz positiva definida (para que sea una parábola bonita)
    dim = 100


    nIters = 2
    tol = 0.0000001
    
    println("Ejecutando versión normal")

    #Creation of the varibles
    #@btime generatorValues_Dense3($dim, 10)
    A, b = generatorValues_Dense3(dim, 10)
    f, g, h = QuadraticFunctions(A, b)  
    println("Here")


    
    # Initialize random vector of same dimension
    #println("Dimension problem:", dim)
    x0 = rand(dim)

    #Run the NAMGM algoritms (It work with the implementation of all methods)
    xf_Oviedo, historial_Ovideo = namgmOviedo(g, h, x0, tol, nIters, modifyHessian_Eigen)
    # xf_Queue, historial_Queue = namgmGrads(G, H, x0, tol, nIters, 8)
    # xf_random, historial_random = namgmRandomVectors(G, H, x0, tol, nIters, 8)
    # xf_newton, historial_newton = newtonMethod(G, H, x0, tol, nIters)


end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end