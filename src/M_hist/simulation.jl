module Simulation

using Random, ProgressMeter, Statistics, Distributed, SharedArrays

include("network.jl")

# const DEFAULT_EPSILON = 0.01  # Define epsilon as a constant at the top of the code

# Decision function f(z)
function decision_function(Zj::Vector{Float64}, alpha::Float64)::Vector{Float64}
    return alpha .* (Zj .- 0.5) .+ 0.5
end

function caluculate_energy(N::Int, X::Vector{Int}, h::Float64, J::Float64)::Float64
    X_spin = 2 * X .- 1
    Energy = - sum(h * X_spin)
    for i in 1:N, j in 1:N
        if i != j
            Energy -= J *  X_spin[i] * X_spin[j] * (1/(N-1))
        end
    end
    return Energy
end

function caluclate_pheromone(S::Vector{Float64}, E::Vector{Float64}, r::Int, t::Int)::Float64
    return S[t-1] + exp(-E[t])
end

function caluclate_pheromone(Sk::Vector{Float64}, E::Vector{Float64}, X::Vector{Int}, r::Int, t::Int)::Vector{Float64}
    return Sk .+ exp(-E[t]) .* X
end

# Initialize the simulation up to the initial time r.
function initialize_simulation(N::Int, X::Matrix{Int}, S::Vector{Float64}, Sk::Vector{Float64}, E::Vector{Float64}, r::Int, h::Float64, J::Float64)
    for t in 1:(r + 1)
        X[:, t] .= rand(0:1, N)
        E[t] = caluculate_energy(N, X[:, t], h, J)
        S[t] = (t == 1 ? E[t] : caluclate_pheromone(S, E, r, t))
        Sk .= (t == 1 ? E[t] .* X[:, t] : caluclate_pheromone(Sk, E, X[:, t], r, t))
    end
end

# Main simulation function
function simulate_ants(N::Int, T::Int, r::Int, omega::Float64, alpha::Float64, h::Float64, J::Float64, progressBar::ProgressMeter.Progress)
    X = zeros(Int, N, T)
    Sk = zeros(Float64, N)
    S = zeros(Float64, T)
    E = zeros(Float64, T)
    Zk = zeros(Float64, N)
    M = zeros(Float64, N)

    # network_popularity関数からk_out配列を取得
    _, _, link_matrix = Network.generate_network(T, r, omega)

    # Initialization
    initialize_simulation(N, X, S, Sk, E, r, h, J)

    # Main loop
    for t in (r + 2):T
        Zk = Sk ./ S[t-1]
        prob = decision_function(Zk, alpha)
        rand_values = rand(Float64, N)
        X[:, t] .= rand_values .< prob
        E[t] = caluculate_energy(N, X[:, t], h, J)

        # S(t) の更新
        linked_ants = filter(x -> x > 0, link_matrix[t, :])  # tにリンクしているアリの選択
        S[t] = sum(exp.(-E[linked_ants]))

        # S(k, t) の更新
        Sk = sum(X[:, linked_ants] .* exp.(-E[linked_ants])', dims=2)[:]
        next!(progressBar)
    end
    M = 2 * alpha * (Zk .- 0.5)
    # Z = S ./ (r * N)

    return M

end

# Function to sample Z values
function sample_ants(N::Int, T::Int, r::Int, omega::Float64, alpha::Float64, h::Float64, J::Float64, samples::Int)
    Z_samples = SharedArray{Float64}(N, samples)

    roop_num = T - (r + 1)
    progressBar = Progress(samples * roop_num, 1, "Samples: ")
    ProgressMeter.update!(progressBar, 0)

    @sync @distributed for i in 1:samples
        Z_samples[:, i] = simulate_ants(N, T, r, omega, alpha, h, J, progressBar)
    end

    #println("Finished simulation")

    M_vector = vcat(Z_samples...)

    return M_vector
end

end