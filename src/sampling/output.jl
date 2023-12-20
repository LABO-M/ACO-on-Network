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
function save_Z_to_csv(Zj_last::Vector{Float64}, filename::String)
    df = DataFrame(Zj=Zj_last)
    CSV.write(filename, df)
    println("Saved Zj values to $filename")
end
