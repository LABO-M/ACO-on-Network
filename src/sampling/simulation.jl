module Simulation

using Random, ProgressMeter

include("network.jl")

const DEFAULT_EPSILON = 0.01  # Define epsilon as a constant at the top of the code

# Decision function f(z)
function decision_function(Zj::Vector{Float64}, alpha::Float64, epsilon::Float64)::Vector{Float64}
    return (1 - epsilon) * ((Zj.^alpha) ./ (Zj.^alpha .+ (1 .- Zj).^alpha)) .+ 0.5 * epsilon
end

# Initialize the simulation up to the initial time r.
function initialize_simulation(N::Int, X::Vector{Int}, S::Vector{Float64}, Sj::Vector{Float64}, r::Int, k_out::Vector{Int})
    for t in 1:r
        X .= rand(0:1, N)
        TP = sum(X)
        S[t] = (t == 1 ? TP : S[t-1] + TP * k_out[t])
        Sj .+= X .* TP * k_out[t]
    end
end

# Main simulation function
function simulate_ants(N::Int, T::Int, r::Int, w::Float64, alpha::Float64)
    X = zeros(Int, N)
    Sj = zeros(Float64, N)
    S = zeros(Float64, T + r)

    # network_popularity関数からk_out配列を取得
    k_out = Network.generate_network(T, r, w)

    # Initialization
    initialize_simulation(N, X, S, Sj, r, k_out)

    for t in (r + 1):(r + T)
        Zj = Sj ./ S[t-1]
        prob = decision_function(Zj, alpha, DEFAULT_EPSILON)
        X .= rand(Float64, N) .< prob
        TP = sum(X)
        S[t] += S[t-1] * TP * k_out[t]
        Sj .+= X .* TP * k_out[t]
    end

    Z = S ./ (r * N)

    return Z

end

end
