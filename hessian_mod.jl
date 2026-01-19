using LinearAlgebra

function modifyHessian_Eigen(hessian, epsilon::Float64 = 1e-12)
    """Shift of the matrix to being posive definite."""
    H = Symmetric(hessian)
    #val_max = eigmax(H)
    val_min = eigmin(H)
    val_min < 0.0 ? hessian += (abs(val_min) + epsilon)*I : nothing
    return H
end

function notModifierHessian(hessian::Matrix, epsilon::Float64 = 1e-12)
    """As its name suggest, this does not realice any transformation to the hessian."""
    return Symmetric(hessian)
end

function diagonalModifier_Hessian(hessianMatrix::Matrix,  epsilon::Float64 = 1e-12)
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


function tridiagonalModifier_Hessian(hessianMatrix::Matrix, epsilon::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix plus an epsilon- if it's
    needed.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - dim:           Int       - Dimension of the matrix    
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Outpu:
        -h: Tridiaognal - The tridiagonal of the approximation of the hesssian.        
        """
    h = Tridiagonal(hessianMatrix) + epsilon*I
    return Symmetric(h)
end

   
function BB_Aproximattion(s::Vector,y::Vector)
    a = (s' * s)/(s' * y)
    I_n = Matrix{Float32}(I, length(s), length(s))
    return a*I_n
end