# =========================
# 厳密 APL（全ペア最短距離の平均）の時系列 @ Julia + Graphs.jl
# - ω = -1（格子）：数式で厳密＆高速
# - ω ≠ -1（一般）：Graphs.jl で全頂点Dijkstra合算 → 厳密
# =========================

using Random, StatsBase, Statistics
using Graphs
using Base.Threads
using Printf
# プロットが必要なら有効化（初回は ] add Plots）
using Plots

# -------------------------
# 生成器：無向隣接リストを直接構築（あなたの元コードをほぼ踏襲）
# -------------------------
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

# -------------------------
# Graphs.jl 用：隣接リスト(1..t) → SimpleGraph
# -------------------------
function to_graph_prefix(adj::AbstractVector{<:AbstractVector{Int32}}, t::Int)
    g = SimpleGraph(t)
    @inbounds for v in 1:t
        for u32 in adj[v]
            u = Int(u32)
            if 1 <= u <= t && u > v
                add_edge!(g, v, u)
            end
        end
    end
    return g
end

# -------------------------
# 厳密APL（一般無向）：全頂点から Dijkstra（=無重みなのでBFS距離）を回して合算
# ・距離行列は保持しない → メモリ O(n+|E|)
# ・未到達は typemax(T) 扱い → 除外
# ・無向の重複（s→v と v→s）は最後に /2
# -------------------------
function exact_apl_graph(g::SimpleGraph)
    n = nv(g)

    # 単スレッドのときは逐次（最も安全）
    if Threads.nthreads() == 1
        total_sum = 0.0
        total_cnt = 0.0
        for s in 1:n
            ds = dijkstra_shortest_paths(g, s).dists  # 無重み=単位重み → BFS距離と同じ
            @inbounds for v in 1:n
                if v == s; continue; end
                d = ds[v]
                if isfinite(d)                      # 未到達=Infを除外（Graphs.jlの距離はFloat64）
                    total_sum += d
                    total_cnt += 1.0
                end
            end
        end
        return (total_sum/2) / (total_cnt/2)        # 無向の重複を/2で調整
    end

    # マルチスレッド時はスレッド別部分和→最後に合算
    sums   = fill(0.0, Threads.nthreads())
    counts = fill(0.0, Threads.nthreads())

    Threads.@threads for s in 1:n
        ds = dijkstra_shortest_paths(g, s).dists
        local_sum = 0.0
        local_cnt = 0.0
        @inbounds for v in 1:n
            if v == s; continue; end
            d = ds[v]
            if isfinite(d)
                local_sum += d
                local_cnt += 1.0
            end
        end
        tid = Threads.threadid()
        @inbounds begin
            sums[tid]   += local_sum
            counts[tid] += local_cnt
        end
    end

    total_sum = sum(sums)   / 2.0
    total_cnt = sum(counts) / 2.0
    return total_sum / total_cnt
end


# -------------------------
# 厳密APL（格子：Path^r） dist(i,j)=ceil(|i-j|/r)
# Σ_{d=1}^{t-1} (t-d) * ceil(d/r) をブロックで O(t/r) 計算
# -------------------------
function exact_apl_lattice(t::Int, r::Int)
    total = 0.0
    K = cld(t-1, r)  # ceil((t-1)/r)
    @inbounds for k in 1:K
        d1 = (k-1)*r + 1
        d2 = min(k*r, t-1)
        m  = d2 - d1 + 1
        s_d = (d1 + d2) * m / 2           # Σd
        total += k * (m*t - s_d)          # Σ(t-d)*ceil(d/r)
    end
    pairs = t*(t-1)/2
    return total / pairs
end

# -------------------------
# 時系列（厳密）
# omega == -1.0 → 格子の厳密式
# それ以外 → Graphs.jl で厳密計算
# 戻り値: Vector{Tuple{Int,Float64}} = (t, apl)
# -------------------------
function apl_timeseries_exact(adj::Vector{Vector{Int32}}, checkpoints::Vector{Int};
                              r::Int, omega::Float64)
    out = Vector{Tuple{Int,Float64}}(undef, length(checkpoints))
    if omega == -1.0
        @inbounds for (i,t) in pairs(checkpoints)
            out[i] = (t, exact_apl_lattice(t, r))
            @info(@sprintf "Exact APL [lattice] t=%d : %.6f" t out[i][2])
        end
    else
        @inbounds for (i,t) in pairs(checkpoints)
            @info "Building graph for t=$t ..."
            g = to_graph_prefix(adj, t)
            @info "Computing exact APL at t=$t with $(nv(g)) nodes and $(ne(g)) edges (threads=$(nthreads())) ..."
            apl = exact_apl_graph(g)
            out[i] = (t, apl)
            @info(@sprintf "Exact APL [general] t=%d : %.6f" t apl)
        end
    end
    return out
end

# ===== チェックポイント生成：前半は細かく、後半は対数スケール =====
"""
front_heavy_checkpoints(Tmax; r, t_small, step_small, log_points, include_Tmax)

- 前半: 1..t_small を `step_small` 刻み（線形）で高解像度
- 後半: t_small..Tmax を `log_points` 個の対数等間隔
- r: 最低でも r+1 以上から始める（初期化の都合）
- include_Tmax: 最後に Tmax を必ず含める
"""
function front_heavy_checkpoints(Tmax::Int;
    r::Int,
    t_small::Int        = 50_000,    # 前半の“細かく見る”上限
    step_small::Int     = 1_000,     # 前半の刻み幅
    log_points::Int     = 40,        # 後半（大スケール側）の点の個数
    include_Tmax::Bool  = true
)
    tmin = max(r + 1, 2)                          # 安全な最小 t
    t_small = min(max(t_small, tmin), Tmax)        # 範囲クリップ
    # 前半: 線形
    cp_lin = collect(tmin:step_small:t_small)

    # 後半: 対数（t_small 以上のみ）
    cp_log = (t_small < Tmax) ? begin
        xs = exp.(range(log(t_small), log(Tmax), length=log_points))
        round.(Int, xs)
    end : Int[]

    cps = vcat(cp_lin, cp_log)
    if include_Tmax && (isempty(cps) || cps[end] != Tmax)
        push!(cps, Tmax)
    end
    # クリーンアップ：範囲内 & 重複除去 & ソート
    cps = filter(t -> tmin <= t <= Tmax, cps)
    unique!(cps)
    sort!(cps)
    return cps
end

# -------------------------
# メイン：パラメータと実行
# ※ まずは小さめの Tmax で検証してください（一般グラフは超重い）
# -------------------------

# ===== 実行パラメータ =====
const Tmax  = 300_000          # ← 一般グラフの厳密は超重いので最初は 1e5 程度で検証
const r     = 100
const omega = -0.9999            # -1.0 で格子（式で即時厳密）、他は人気モデル（厳密は重い）
const seed  = 42

# チェックポイント（例：線形間隔）
checkpoints = front_heavy_checkpoints(Tmax; r=r,
    t_small=50_000,    # 前半を細かく見る上限
    step_small=100,  # その刻み
    log_points=30,     # 後半は40点を対数等間隔
    include_Tmax=true
)
@info "num checkpoints = $(length(checkpoints))"
# 格子(ω=-1)なら 1_000_000 でもOK。一般(≠-1)は計算資源に応じて間引きを推奨。

# ===== ネットワーク生成 =====
@info "Generating network Tmax=$Tmax, r=$r, ω=$omega ..."
adj = [Int32[] for _ in 1:Tmax]
p = NetParams(Int32(Tmax), Int32(r), omega)
generate_network!(adj, p; seed=seed)
@info "Network generated."

# ===== 厳密APLの時系列 =====
res = apl_timeseries_exact(adj, checkpoints; r=r, omega=omega)

# ===== 出力（テーブル） =====
println("t, exact_apl")
for (t, apl) in res
    @printf("%d, %.6f\n", t, apl)
end

# ===== プロット（必要なら） =====
ts  = [t  for (t, apl) in res]
aps = [apl for (t, apl) in res]
plt = plot(ts, aps; xlabel="t", ylabel="average path length", title="Exact APL time series (r=$(r), ω=$(omega))", legend=false)
display(plt)
savefig(plt, "apl_timeseries_exact.png")
println("Saved plot: apl_timeseries_exact.png")
