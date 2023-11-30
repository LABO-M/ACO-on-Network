module Network

using Random, ProgressMeter, StatsBase

function initialize_network(T::Int, r::Int)
    k_in = zeros(Int, T)
    k_out = zeros(Int, T)

    for t in 1:(r + 1)
        k_in[t] = r
        k_out[t] = (r + 1) - t
    end

    return k_in, k_out
end

function generate_network(T::Int, r::Int, w::Float64)
    if w == -1.0
        return network_lattice(T, r)
    else
        return network_popularity(T, r, w)
    end
end

function network_popularity(T::Int, r::Int, w::Float64)
    # 初期条件の設定
    k_in, k_out = initialize_network(T, r)
    l = zeros(Float64, T)

    progressBar = Progress(T, 1, "Evoluting network")

    # ネットワークの進化
    for t in (r + 2):T
        k_in[t] = r
        # 人気度の更新（負の値を0に置き換える）
        l .= max.(0, k_in .+ w .* k_out)

        # 人気度が0より大きいアリを選択可能なリストに追加
        popular_ants = findall(x -> x > 0, l)

        # 人気度の合計
        total_popularity = sum(l[popular_ants])

        # 既存のノードから r または r' アリを選択
        num_links = min(r, length(popular_ants))

        # 重み付きサンプリングでアリを選択
        probabilities = l[popular_ants] / total_popularity
        selected_ants = sample(popular_ants, Weights(probabilities), num_links, replace=false)

        # 選ばれたアリの k_out を更新
        k_out[selected_ants] .+= 1
        next!(progressBar)
    end

    return k_out
end

function network_lattice(T::Int, r::Int)
    # 初期条件の設定
    k_in, k_out = initialize_network(T, r)

    # ネットワークの進化
    for t in (r + 2):T
        k_in[t] = r

        # 既存のノードの出次数を更新
        k_out[1:max(1, t - r - 1)] .= r
        if t - r > 0
            k_out[(t - r):(t - 1)] .= (r - 1):-1:0
        end
    end

    return k_out
end

end
