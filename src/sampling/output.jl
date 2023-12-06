using CSV, DataFrames

# Convert the numeric value to a string format with K, M, G, etc.
function int_to_SI_prefix(value::Int)
    if value % 1_000_000_000 == 0
        return string(value ÷ 1_000_000_000) * "G"
    elseif value % 1_000_000 == 0
        return string(value ÷ 1_000_000) * "M"
    elseif value % 1_000 == 0
        return string(value ÷ 1_000) * "K"
    else
        return string(value)
    end
end

# Function to save Z values to a CSV file with optional downsampling
function save_Z_to_csv(Z::Vector{Float64}, filename::String)
    t_values = collect(1:length(Z))
    df = DataFrame(t=t_values, Z=Z)
    CSV.write(filename, df)
    println("Saved Z values to $filename")
end

# Function to save the link matrix to a CSV file
function save_link_matrix_to_csv(link_matrix::Matrix{Int}, filename::String)
    df = DataFrame(link_matrix, :auto)
    CSV.write(filename, df)
    println("Saved link matrix to $filename")
end
