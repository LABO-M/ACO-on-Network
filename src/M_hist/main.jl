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
        default = 100
        arg_type = Int

        "--omega"
        help = "weight parameter of the network"
        default = -0.9999
        arg_type = Float64

        "--alpha"
        help = "Exponent parameter alpha"
        default = 0.1
        arg_type = Float64

        "--h"
        help = "External field h"
        default = 0.001
        arg_type = Float64

        "--J"
        help = "Coupling constant J"
        default = 0.1
        arg_type = Float64

        "--sample"
        help = "Sample size."
        default = 100
        arg_type = Int

    end

    parsed_args = parse_args(args, s)
    N = parsed_args["N"]
    T = parsed_args["T"]
    r = parsed_args["r"]
    omega = parsed_args["omega"]
    alpha = parsed_args["alpha"]
    h = parsed_args["h"]
    J = parsed_args["J"]
    samples = parsed_args["sample"]

    # Log the simulation parameters
    #println("Running simulation with the following parameters:")
    #println("N = $(int_to_SI_prefix(N)), T = $(int_to_SI_prefix(T)), r = $(int_to_SI_prefix(r)), omega = $(omega), alpha = $(alpha), h = $(h), J = $(J) samples = $(int_to_SI_prefix(samples))")

    # Run the simulation
    M_vector = Simulation.sample_ants(N, T, r, omega, alpha, h, J, samples)

    # Output Z values to CSV
    dir_Z = "data/ising/Zt"
    if !isdir(dir_Z)
        mkpath(dir_Z)
    end
    filename_Z = joinpath(dir_Z, "N$(int_to_SI_prefix(N))_T$(int_to_SI_prefix(T))_r$(int_to_SI_prefix(r))_omega$(omega)_alpha$(alpha)_h$(h)_J$(J).csv")
    save_Z_to_csv(M_vector, filename_Z)
    end

# Entry point of the script
isinteractive() || main(ARGS)
