module Simulation

using Random, ProgressMeter, Statistics, Distributed, SharedArrays

include("network.jl")

const DEFAULT_EPSILON = 0.01  # Define epsilon as a constant at the top of the code

# Decision function f(z)
function decision_function(Zj::Vector{Float64}, alpha::Float64, epsilon::Float64)::Vector{Float64}
    return (1 - epsilon) * ((Zj.^alpha) ./ (Zj.^alpha .+ (1 .- Zj).^alpha)) .+ 0.5 * epsilon
end

# Initialize the simulation up to the initial time r.
function initialize_simulation(N::Int, X::Matrix{Int}, S::Vector{Float64}, Sj::Vector{Float64}, TP::Vector{Int}, r::Int)
    for t in 1:(r + 1)
        X[:, t] .= rand(0:1, N)
        TP[t] = sum(X[:, t])
        S[t] = (t == 1 ? TP[t] : S[t-1] + TP[t])
        Sj .+= X[:, t] .* TP[t]
    end
end

# Main simulation function
function simulate_ants(N::Int, T::Int, r::Int, omega::Float64, alpha::Float64, progressBar::ProgressMeter.Progress)
    X = zeros(Int, N, T)
    Sj = zeros(Float64, N)
    S = zeros(Float64, T)
    TP = zeros(Int, T)

    # network_popularity関数からk_out配列を取得
    _, _, link_matrix = Network.generate_network(T, r, omega)

    # Initialization
    initialize_simulation(N, X, S, Sj, TP, r)

    # Main loop
    for t in (r + 2):T
        Zj = Sj ./ S[t-1]
        prob = decision_function(Zj, alpha, DEFAULT_EPSILON)
        rand_values = rand(Float64, N)
        X[:, t] .= rand_values .< prob
        TP[t] = sum(X[:, t])

        # S(t) の更新
        linked_ants = filter(x -> x > 0, link_matrix[t, :])  # tにリンクしているアリの選択
        S[t] = sum(TP[linked_ants])

        # S(j, t) の更新
        Sj = sum(X[:, linked_ants] .* TP[linked_ants]', dims=2)[:]
        next!(progressBar)
    end

    Z = S ./ (r * N)

    return Z

end

# Function to sample Z values
function sample_ants(N::Int, T::Int, r::Int, omega::Float64, alpha::Float64, samples::Int)::Tuple{Vector{Float64}, Vector{Float64}}
    Z_samples = SharedArray{Float64}(T, samples)

    roop_num = T - (r + 1)
    progressBar = Progress(samples * roop_num, 1, "Samples: ")
    ProgressMeter.update!(progressBar, 0)

    @sync @distributed for i in 1:samples
        Z_samples[:, i] = simulate_ants(N, T, r, omega, alpha, progressBar)
    end

    println("Finished simulation")

    # Calculate mean and standard deviation values
    Z_mean = mean(Z_samples, dims=2)
    Z_std = std(Z_samples, dims=2)

    return vec(Z_mean), vec(Z_std)
end

end
