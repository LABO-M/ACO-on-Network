include("network.jl")

T = 100000
r = 100
omega = -0.99

@time Network.generate_network(T, r, omega)
