# =========================
# 100万ノード向け：平均距離の時系列推定（無向）
# =========================
using Random, StatsBase, Statistics

# ---- 無向隣接リストを直接構築する生成器 ----
struct NetParams
    T::Int32
    r::Int32
    omega::Float64  # -1.0 なら格子、それ以外は人気モデル
end

"""
generate_network!(adj, p; seed)

adj::Vector{Vector{Int32}} を直接埋める（無向）
p.omega == -1 なら lattice、それ以外は popularity
"""
function generate_network!(adj::Vector{Vector{Int32}}, p::NetParams; seed=0)
    Random.seed!(seed)
    T = Int(p.T); r = Int(p.r); ω = p.omega

    # in/out 度（人気度計算に使用）
    k_in  = zeros(Int32, T)
    k_out = zeros(Int32, T)
    l     = zeros(Float64, T)  # 人気度

    # 初期化（あなたの initialize_network と同等の効果）
    k_out[1] = r
    for t in 2:(r+1)
        k_in[t] = t-1
        k_out[t] = r - (t-1)
        # t が 1..(t-1) にリンクする → 無向で両側に追加
        for parent in 1:(t-1)
            push!(adj[t], Int32(parent))
            push!(adj[parent], Int32(t))
        end
    end
    l .= k_in .+ ω .* k_out

    if ω == -1.0
        # ---- 格子モデル ----
        for t in (r+2):T
            # t は (t-r)..(t-1) に接続
            a = max(1, t-r)
            for parent in a:(t-1)
                push!(adj[t], Int32(parent))
                push!(adj[parent], Int32(t))
            end
            # 度更新（見た目が合うように）
            k_in[t] = r
            k_out[1:max(1, t - r - 1)] .= r
            if t - r > 0
                k_out[(t - r):(t - 1)] .= Int32.((r - 1):-1:0)
            end
        end
    else
        # ---- 人気モデル ----
        for t in (r+2):T
            # 人気度>0 の候補
            # （ベクトル走査は重いので、r が小さければこのままでも実用）
            popular = findall(@view(l .> 0.0))
            if isempty(popular)
                continue
            end
            total_pop = sum(@view l[popular])
            num_links = min(r, length(popular))
            probs = (@view l[popular]) ./ total_pop
            selected = sample(popular, Weights(probs), num_links; replace=false)

            # エッジ追加（無向）
            for parent in selected
                push!(adj[t], Int32(parent))
                push!(adj[parent], Int32(t))
                k_out[parent] += 1
                k_in[t] += 1
            end
            # 人気度更新（差分で十分だが簡潔に全体更新）
            @inbounds begin
                for v in selected
                    l[v] = k_in[v] + ω*k_out[v]
                end
                l[t] = k_in[t] + ω*k_out[t]
            end
        end
    end

    # 重複除去（同一親を複数回引いた場合などの保険）
    for v in 1:T
        if length(adj[v]) > 1
            adj[v] = unique(adj[v])
        end
    end
    return nothing
end

# ---- 高速BFS（配列再利用型、到達不能は -1）----
mutable struct BFSWorkspace
    dist::Vector{Int32}
    q::Vector{Int32}
end

function BFSWorkspace(n::Int)
    BFSWorkspace(fill(Int32(-1), n), Vector{Int32}(undef, n))
end

@inline function bfs!(adj::Vector{Vector{Int32}}, s::Int32, ws::BFSWorkspace)
    dist = ws.dist; q = ws.q
    n = length(dist)
    # reset dist to -1 (高速化のため fill! は十分速い)
    fill!(dist, Int32(-1))
    head = 1; tail = 0
    dist[s] = 0
    tail += 1; q[tail] = s
    @inbounds while head <= tail
        v = q[head]; head += 1
        dv = dist[v] + 1
        for u in adj[v]
            if dist[u] == -1
                dist[u] = dv
                tail += 1; q[tail] = u
            end
        end
    end
    return dist
end

# ---- サンプリング平均距離（単一点→他点の平均）----
@inline function mean_distance_from!(adj, s::Int32, ws::BFSWorkspace)
    dist = bfs!(adj, s, ws)
    tot::Int64 = 0
    cnt::Int64 = 0
    @inbounds for (t, d) in enumerate(dist)
        if t == s; continue; end
        if d >= 0
            tot += d
            cnt += 1
        end
    end
    return cnt == 0 ? NaN : (tot / cnt)
end

"""
estimate_apl!(adj; S)

ランダムに S 個の始点を取り、各始点の「他ノードまでの平均距離」を平均。
戻り値: (μ, se)  推定平均と標準誤差
"""
function estimate_apl!(adj::Vector{Vector{Int32}}; S::Int=256)
    n = length(adj)
    ws = BFSWorkspace(n)
    samples = Float64[]
    for s in rand(1:n, S)
        m = mean_distance_from!(adj, Int32(s), ws)
        if !isnan(m)
            push!(samples, m)
        end
    end
    μ = mean(samples)
    se = std(samples) / sqrt(length(samples))
    return μ, se
end

"""
apl_timeseries(Tmax, r, omega; checkpoints, S, seed)

checkpoints: 計測ノード数の配列（例: 対数間隔）
S: サンプリング始点数
"""
function apl_timeseries(Tmax::Int, r::Int, omega::Float64;
                        checkpoints::Vector{Int}=floor.(Int, unique!(round.(exp.(range(log(100), log(Tmax), length=20))))),
                        S::Int=256, seed::Int=0)

    # 隣接リスト（1..Tmax）を先に確保しておき、途中まで使う
    adj = [Int32[] for _ in 1:Tmax]
    p = NetParams(Int32(Tmax), Int32(r), omega)
    generate_network!(adj, p; seed=seed)

    results = Vector{Tuple{Int,Float64,Float64}}()
    for t in checkpoints
        # 1..t の部分グラフで評価（ビュー的には adj[1:t] を渡せばOK）
        μ, se = estimate_apl!(@view adj[1:t]; S=S)
        push!(results, (t, μ, se))
        @info "APL at t=$t: $(round(μ, digits=4)) ± $(round(se, digits=4))"
    end
    return results
end

# ===== 実行例 =====
const Tmax  = 1_000_000
const r     = 100
const omega = -0.99      # -1.0 で格子
const S     = 256
# 対数間隔のチェックポイント（必要なら密に）
checkpoints = floor.(Int, unique!(round.(exp.(range(log(1_000), log(Tmax), length=18)))))

res = apl_timeseries(Tmax, r, omega; checkpoints=checkpoints, S=S, seed=42)

# 出力
println("t, est_apl, se")
for (t, μ, se) in res
    println("$(t), $(round(μ, digits=4)), $(round(se, digits=4))")
end
