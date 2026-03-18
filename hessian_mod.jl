using LinearAlgebra

function modifyHessian_Eigen(hessianMatrix, gradient = Nothing,  epsilon::Float64 = 1e-12)
    """Function that shifts the eigenvalues of the hessian matrix to make it positive definite and
    an epsilon is added- if it's stated. This is the implementation of the Example 4.3.6 of 
    my masters thesis. The code name in get_modifier is 'eigen'.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - gradient:  Vector/Array  - Gradient of the function in the present iteration
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Output:
        -h: Diagonal - The main diagonal of the approximation of the hesssian.        
    """
    h = Symmetric(hessianMatrix)
    #val_max = eigmax(H)
    val_min = eigmin(h)
    val_min < 0.0 ? h += (abs(val_min) + epsilon)*I : nothing
    return h
end

function notModifierHessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
    """As its name suggest, this does not realice any transformation to the hessian.
    The code name in get_modifier is 'none'"""
    return Symmetric(hessianMatrix)
end

#--------------------------------------- Diagonal Family ---------------------------------------#
function diagonalModifier_Hessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix and square the elements plus 
    an epsilon- if it's stated. This is the implementation of the Example 4.3.4 of 
    my masters thesis. The code name in get_modifier is 'diag'.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - gradient:  Vector/Array  - Gradient of the function in the present iteration
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Output:
        -h: Diagonal - The main diagonal of the approximation of the hesssian.        
    
    # Remark:
        This is not a good choice because it doubles the condition number of the resulting approximation.
        But still is an example.
    """
    h = Diagonal(diag(hessianMatrix)).^2 + epsilon*I
    return Symmetric(h)
end

function diagonalModifier_sabs_Hessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix take the absolut value
    and sum an epsilon- if it's stated. This is the implementation of the Example 4.3.5 of 
    my masters thesis. The code name in get_modifier is 'sabsdiag'.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - gradient:  Vector/Array  - Gradient of the function in the present iteration
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Output:
        -h: Diaognal - The main diagonal of the approximation of the hesssian.       
    
    # Remark:
        Sabs comes from 'Squared Absolute Value'.
    """
    d = Diagonal(sqrt.(abs.(diag(hessianMatrix)))) + epsilon*I 
    return Symmetric(d)
end

function diagonalModifier_max_Hessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix, then compute the minimal eigenvalue
    and makes checks if's necessary to add a value to make it positive definite.This is the implementation 
    of the Example 4.3.6 of my masters thesis. The code name in get_modifier is 'maxdiag'.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - gradient:  Vector/Array  - Gradient of the function in the present iteration
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Output:
        -h: Diaognal - The main diagonal of the approximation of the hesssian.       
    
    # Remark:
        In some sense this approximation is the analogue of the modifyHessian Routine just that for
        diagonal matrices.
    """
    #Compute the minimial eigenvalue
    val_min = eigmin(Symmetric(hessianMatrix))
    
    #Construct the lambda value
    lambda = epsilon + maximum([0, -val_min])
    
    #Construct the approximation of the hessian 
    h = Diagonal(diag(hessianMatrix)) + lambda*I

    return h
end

#-----------------------------------------------------------------------------------------------#

function tridiagonalModifier_Hessian(hessianMatrix::Matrix, gradient = Nothing, epsilon::Float64 = 1e-12)
    """Function that gets the main diagonal of the Hessian Matrix plus an epsilon- if it's
    needed. The code name in get_modifier is 'tridiag'.
    
    # Inputs:
        - hessianMatrix: Symmetric - Approximation of the hessian matrix
        - gradient:  Vector/Array  - Gradient of the function in the present iteration
        - epsilon:       Float64   - A small real number to prevent the hessian to be the 0 matrix.

    # Outpu:
        -h: Tridiaognal - The tridiagonal of the approximation of the hesssian.        
        """
    h = Tridiagonal(hessianMatrix) + epsilon*I
    return Symmetric(h)
end


function removeConvergenceModifier(hessian, gradient = nothing, epsilon::Float64 = 0.0)
    """Function that theoretically removes the convergence of the NAMGM. Making the Hessian matrix
    into an approximation Bk such that Bkgk = 0 for all k, using only a diagonal matrix.  This is the 
    implementation of the Example 4.3.3 of my masters thesis. The code name in get_modifier is 'remove'.
    
    # Inputs: 
        - Hessian: Matrix - Hessian on an iteration
        - gradient:  Vector/Array  - Gradient of the function in the present iteration
        - Epsion: Float64 - Small value to shift the matrix if it's needed
    
    # Output:
        - Bk: Matrix - Approximation of a hessian matrix making it a bad choice for the method

    # Remarks:
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
        #addedMatrix[i,i] = isapprox(g, 0.0) ? 0.0 : -1/g * Hkgk[i]
    end
    return Symmetric(hessian + addedMatrix)
end