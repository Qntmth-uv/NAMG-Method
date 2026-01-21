using LinearAlgebra

function modifyHessian_Eigen(hessian, gradient = Nothing,  epsilon::Float64 = 1e-12)
    """Shift of the matrix to being posive definite."""
    H = Symmetric(hessian)
    #val_max = eigmax(H)
    val_min = eigmin(H)
    val_min < 0.0 ? hessian += (abs(val_min) + epsilon)*I : nothing
    return H
end

function notModifierHessian(hessian::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
    """As its name suggest, this does not realice any transformation to the hessian."""
    return Symmetric(hessian)
end

function diagonalModifier_Hessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
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


function tridiagonalModifier_Hessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
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


function removeConvergenceModifier(hessian, gradient = nothing, epsilon::Float64 = 0.0)
    """Function that theoryctly removes the convergence of the NAMGM making the Hessian matrix
    into an approximation Bk such that Bkgk = 0 using only a diagonal matrix
    
    # Input 
        - Hessian: Matrix - Hessian on an iteration
        - Epsion: Float64 - Small value to shift the matrix if it's needed
    
    # Output
        - Bk: Matrix - Approximation of a hessian matrix making it a bad choice for the method

    # Remarks
        - This is only a way to construct such matrix (actually is the easiest way to construct it).
        The other elements in the diagonal matrix can be any choice, for simplicity we take them as 0.0.
    """
    #Init the variable
    dim = length(gradient) 
    addedMatrix = zeros(Float64, dim, dim)
    
    #Result
    Hkgk = hessian*gradient

    #Fill the matrix
    for (i, g) in enumerate(Hkgk)
        addedMatrix[i,i] = -1/g * Hkgk[i] 
        #addedMatrix[i,i] = isapprox(g, 0.0) ? 0.0 : 1/g * Hkgk[i]
    end

    #Return the matrix
    #display(addedMatrix)
    #display(gradient-Hkgk)
    return Symmetric(hessian + addedMatrix)
end