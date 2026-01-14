using DataFrames
using CSV


function getIterationSpeed(historal, time::Float64)
    """Compute the avrg iteration speed."""
    n = length(historal)
    return n/time  
end


function createCSV(arrayofHistorials, CSV_fileName::String, headers )

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
    