# =========================
# 100万ノード向け：平均距離の時系列推定（無向）+ プロット
# =========================
using Random, StatsBase, Statistics
using Plots  # 初回は: ] add Plots

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

    # 初期化（initialize_network 相当）
    k_out[1] = r
    for t in 2:(r+1)
        k_in[t] = t - 1
        k_out[t] = r - (t - 1)
        # t が 1..(t-1) にリンク → 無向で両側に追加
        for parent in 1:(t-1)
            push!(adj[t], Int32(parent))
            push!(adj[parent], Int32(t))
        end
    end
    l .= k_in .+ ω .* k_out

    if ω == -1.0
        # ---- 格子モデル ----
        for t in (r+2):T
            a = max(1, t - r)
            for parent in a:(t - 1)
                push!(adj[t], Int32(parent))
                push!(adj[parent], Int32(t))
            end
            k_in[t] = r
            k_out[1:max(1, t - r - 1)] .= r
            if t - r > 0
                k_out[(t - r):(t - 1)] .= Int32.((r - 1):-1:0)
            end
        end
    else
        # ---- 人気モデル ----
        for t in (r+2):T
            popular = findall(l .> 0.0)
            if isempty(popular)
                continue
            end
            total_pop = sum(@view l[popular])
            num_links = min(r, length(popular))
            probs = (@view l[popular]) ./ total_pop
            selected = sample(popular, Weights(probs), num_links; replace=false)

            for parent in selected
                push!(adj[t], Int32(parent))
                push!(adj[parent], Int32(t))
                k_out[parent] += 1
                k_in[t] += 1
            end
            @inbounds begin
                for v in selected
                    l[v] = k_in[v] + ω*k_out[v]
                end
                l[t] = k_in[t] + ω*k_out[t]
            end
        end
    end

    # 重複除去（保険）
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

BFSWorkspace(n::Int) = BFSWorkspace(fill(Int32(-1), n), Vector{Int32}(undef, n))

# @view adj[1:t] でも動くよう AbstractVector を許容し、
# 部分グラフ外（u>n）をスキップするガードを入れる
@inline function bfs!(adj::AbstractVector{<:AbstractVector{Int32}}, s::Int32, ws::BFSWorkspace)
    dist = ws.dist; q = ws.q
    n = length(dist)
    fill!(dist, Int32(-1))
    head = 1; tail = 0
    dist[Int(s)] = 0
    tail += 1; q[tail] = s
    @inbounds while head <= tail
        v = q[head]; head += 1
        dv = dist[Int(v)] + 1
        for u32 in adj[Int(v)]
            u = Int(u32)
            if u > n || u < 1
                continue
            end
            if dist[u] == -1
                dist[u] = dv
                tail += 1; q[tail] = Int32(u)
            end
        end
    end
    return dist
end

# ---- サンプリング平均距離（単一点→他点の平均）----
@inline function mean_distance_from!(adj::AbstractVector{<:AbstractVector{Int32}}, s::Int32, ws::BFSWorkspace)
    dist = bfs!(adj, s, ws)
    tot::Int64 = 0
    cnt::Int64 = 0
    @inbounds for (t, d) in enumerate(dist)
        if t == Int(s); continue; end
        if d >= 0
            tot += d
            cnt += 1
        end
    end
    return cnt == 0 ? NaN : (tot / cnt)
end

"""
estimate_apl!(adj; S)

ランダムに S 個の始点からの平均距離を取り、全体平均を推定。
戻り値: (μ, se)
"""
function estimate_apl!(adj::AbstractVector{<:AbstractVector{Int32}}; S::Int=256)
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
        μ, se = estimate_apl!(@view adj[1:t]; S=S)
        push!(results, (t, μ, se))
        @info "APL at t=$t: $(round(μ, digits=4)) ± $(round(se, digits=4))"
    end
    return results
end

# ===== 実行パラメータ =====
const Tmax  = 1_000_000          # 例: 小さめで動作確認 → 本番で 1_000_000 へ
const r     = 100
const omega = -0.99            # -1.0 で格子、他は人気モデル
const S     = 128
# チェックポイント（例: 線形間隔）
checkpoints = collect(10_000:10_000:Tmax)

# ===== 実行 =====
res = apl_timeseries(Tmax, r, omega; checkpoints=checkpoints, S=S, seed=42)

# ===== 出力（テーブル） =====
println("t, est_apl, se")
for (t, μ, se) in res
    println("$(t), $(round(μ, digits=4)), $(round(se, digits=4))")
end

# ===== プロット =====
ts  = [t  for (t, μ, se) in res]
mus = [μ  for (t, μ, se) in res]
ses = [se for (t, μ, se) in res]

# リボン（±1標準誤差）付き折れ線
plt = plot(
    ts, mus;
    ribbon = ses,
    xlabel = "t",
    ylabel = "average distance",
    title  = "Time serieses of average distance S=$(S), r=$(r), ω=$(omega))",
    legend = false,
)
display(plt)
savefig(plt, "apl_timeseries.png")   # ファイル保存（同ディレクトリ）
println("Saved plot: apl_timeseries.png")
