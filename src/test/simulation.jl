module Simulation

using Random, Distributed, SharedArrays, Statistics
include("network.jl")  # expects: Network.generate_network(T, r, omega) -> (_, _, link_matrix)

# --- 意思決定 f(z) = (1-α)/2 + α z を安全に ---
decision_function(Z::Vector{Float64}, alpha::Float64) =
    clamp.((1 - alpha) * 0.5 .+ alpha .* Z, 0.0, 1.0)

# --- エネルギー（完全グラフ Ising）O(N) ---
function calculate_energy(N::Int, X::Vector{Int}, h::Float64, J::Float64)::Float64
    s = 0
    @inbounds @simd for i in 1:N
        s += (X[i] == 1 ? 1 : -1)               # σ_i = 2X_i - 1
    end
    # Σ_{i≠j} σ_iσ_j = (Σσ)^2 - N
    return -h * s - (J / (N - 1)) * (s * s - N)
end

# --- 初期化：t=1..r+1 はランダム（S/Skは初期化せずEのみ整える） ---
function initialize!(N::Int, X::Matrix{Int}, E::Vector{Float64}, r::Int, h::Float64, J::Float64)
    tmax = min(size(X, 2), r + 1)
    for t in 1:tmax
        @inbounds @simd for i in 1:N
            X[i, t] = rand() < 0.5 ? 1 : 0
        end
        E[t] = calculate_energy(N, X[:, t], h, J)
    end
end

# --- 1サンプル：最終時刻で全Z>0.5か（Bool） ---
function simulate_once_success(N::Int, T::Int, r::Int, omega::Float64, alpha::Float64, h::Float64, J::Float64;
                               rng = Random.default_rng())::Bool
    X = zeros(Int, N, T)
    E = zeros(Float64, T)
    initialize!(N, X, E, r, h, J)

    # あなたの network.jl からリンク行列
    _, _, link_matrix = Network.generate_network(T, r, omega)

    lastZ = zeros(Float64, N)

    # 本ループ：t = r+2 .. T
    for t in (r + 2):T
        # 親集合（0を除外）
        parents = [x for x in link_matrix[t, :] if x > 0]

        # S と S1 を t の親だけから集計
        if isempty(parents)
            fill!(lastZ, 0.0)  # 念のため
        else
            w = exp.(-E[parents])                  # 長さ = |parents|
            S = sum(w)
            if S <= 0
                fill!(lastZ, 0.0)
            else
                # S1(k) = Σ_s w_s * 1{X[k,s]=1}  → 行列×ベクトルで高速に
                # X[:, parents] (N × |parents|) * w (|parents|)
                S1 = Array(X[:, parents]) * w
                @inbounds @simd for k in 1:N
                    lastZ[k] = S1[k] / S
                end
            end
        end

        # Z → 確率 → X(:,t)
        p = decision_function(lastZ, alpha)
        @inbounds @simd for k in 1:N
            X[k, t] = rand(rng) < p[k] ? 1 : 0
        end

        # E(t)
        E[t] = calculate_energy(N, X[:, t], h, J)
    end

    # 成功判定：直近の Z で全kが0.5超
    return all(z -> z > 0.5, lastZ)
end

# --- main.jl 互換：samples 回の結果（1=成功/0=失敗）をベクトルで返す ---
function sample_ants(N::Int, T::Int, r::Int, omega::Float64, alpha::Float64,
                     h::Float64, J::Float64, samples::Int)
    # 1次元の SharedArray（各サンプルの成否を格納）
    results = SharedArray{Float64}(samples)

    @sync @distributed for i in 1:samples
        ok = simulate_once_success(N, T, r, omega, alpha, h, J; rng = MersenneTwister(i))
        results[i] = ok ? 1.0 : 0.0
    end

    # main.jl 側はベクトルをCSVに保存するので、そのまま返す
    return collect(results)
end

end # module
