using DataFrames

using CSV



function getIterationSpeed(historal, time::Float64)
    """Compute the avrg iteration speed."""
    n = length(historal)
    return n/time  
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