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