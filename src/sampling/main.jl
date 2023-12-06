using ArgParse
include("simulation.jl")
include("output.jl")

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--N"
        help = "Number of questions (quizzes)"
        default = 100
        arg_type = Int

        "--T"
        help = "Total number of time steps"
        default = 10000
        arg_type = Int

        "--r"
        help = "Initial number of ants"
        default = 1000
        arg_type = Int

        "--omega"
        help = "weight parameter of the network"
        default = -0.99
        arg_type = Float64

        "--alpha"
        help = "Exponent parameter alpha"
        default = 1.01 # 1 - epsilon
        arg_type = Float64
    end

    parsed_args = parse_args(args, s)
    N = parsed_args["N"]
    T = parsed_args["T"]
    r = parsed_args["r"]
    omega = parsed_args["omega"]
    alpha = parsed_args["alpha"]

    # Log the simulation parameters
    println("Running simulation with the following parameters:")
    println("N = $(int_to_SI_prefix(N)), T = $(int_to_SI_prefix(T)), r = $(int_to_SI_prefix(r)), omega = $(omega), alpha = $(alpha)")

    # Run the simulation
    Z, link_matrix = Simulation.simulate_ants(N, T, r, omega, alpha)

    # Output Z values to CSV
    dir_Z = "data/Zt"
    dir_link = "data/link_matrix"
    if !isdir(dir_Z)
        mkpath(dir_Z)
    end
    if !isdir(dir_link)
        mkpath(dir_link)
    end
    filename_Z = joinpath(dir_Z, "N$(int_to_SI_prefix(N))_T$(int_to_SI_prefix(T))_r$(int_to_SI_prefix(r))_omega$(omega)_alpha$(alpha).csv")
    save_Z_to_csv(Z, filename_Z)
    filename_link = joinpath(dir_link, "N$(int_to_SI_prefix(N))_T$(int_to_SI_prefix(T))_r$(int_to_SI_prefix(r))_omega$(omega)_alpha$(alpha).csv")
    save_link_matrix_to_csv(link_matrix, filename_link)
    end

# Entry point of the script
isinteractive() || main(ARGS)
