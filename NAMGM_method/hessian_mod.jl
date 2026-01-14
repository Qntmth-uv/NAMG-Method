using LinearAlgebra

function modifyHessian_Eigen(hessian, dimension::Int)
    """Shift of the matrix to being posive definite."""
    H = Symmetric(hessian)
    val_max = eigmax(H)
    val_min = eigmin(H)
    if (val_min<0.0)
        hessian += abs(val_min)*I
        #println("The matrix was shifted")
    end
    #println("Mínimo: $val_min - Máximo: $val_max")
    return H
end

function notModifierHessian(hessian::Matrix, dimension::Int)
    """As its name suggest, this does not realice any transformation to the hessian."""
    return Symmetric(hessian)
end

function diagonalModifier_Hessian(hessianMatrix::Matrix, dim::Int, epsion::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix plus an epsilon- if it's
    needed.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - dim:           Int       - Dimension of the matrix 
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Outpu:
        -h: Diaognal - The main diagonal of the approximation of the hesssian.        
        """
    h = Diagonal(diag(hessianMatrix))
    return Symmetric(h)
end


function tridiagonalModifier_Hessian(hessianMatrix::Matrix, dim::Int, epsion::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix plus an epsilon- if it's
    needed.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - dim:           Int       - Dimension of the matrix    
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Outpu:
        -h: Tridiaognal - The tridiagonal of the approximation of the hesssian.        
        """
    h = Tridiagonal(hessianMatrix) + epsion*I
    return Symmetric(h)
end

   
function BB_Aproximattion(s::Vector,y::Vector)
    a = (s' * s)/(s' * y)
    I_n = Matrix{Float32}(I, length(s), length(s))
    return a*I_n
end