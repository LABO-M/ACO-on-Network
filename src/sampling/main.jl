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
        default = -0.99
        arg_type = Float64

        "--alpha"
        help = "Exponent parameter alpha"
        default = 1.0
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
    samples = parsed_args["sample"]

    # Log the simulation parameters
    println("Running simulation with the following parameters:")
    println("N = $(int_to_SI_prefix(N)), T = $(int_to_SI_prefix(T)), r = $(int_to_SI_prefix(r)), omega = $(omega), alpha = $(alpha), samples = $(int_to_SI_prefix(samples))")

    # Run the simulation
    Z_mean, Z_std = Simulation.sample_ants(N, T, r, omega, alpha, samples)

    # Output Z values to CSV
    dir_Z = "data/ising/Zt"
    if !isdir(dir_Z)
        mkpath(dir_Z)
    end
    filename_Z = joinpath(dir_Z, "N$(int_to_SI_prefix(N))_T$(int_to_SI_prefix(T))_r$(int_to_SI_prefix(r))_omega$(omega)_alpha$(alpha).csv")
    save_Z_to_csv(Z_mean, Z_std, filename_Z)
    end

# Entry point of the script
isinteractive() || main(ARGS)
