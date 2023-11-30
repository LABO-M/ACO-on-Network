using ArgParse
include("simulation.jl")
include("output.jl")

function main(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--N"
        help = "Number of questions (quizzes)"
        default = 10
        arg_type = Int
        "--T"
        help = "Total number of time steps"
        default = 1000
        arg_type = Int
        "--r"
        help = "Initial number of ants"
        default = 3
        arg_type = Int
        "--omega"
        help = "weight parameter of the network"
        "--alpha"
        default = - 0.99
        arg_type = Float64
        help = "Exponent parameter alpha"
        default = 0.99 # 1 - epsilon
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
    Z = simulate_ants(N, T, r, omega, alpha)

    # Output Z values to CSV
    dir_Z = "data/Zt"
    if !isdir(dir_Z)
        mkpath(dir_Z)
    end
    filename_Z = joinpath(dir_Z, "N$(int_to_SI_prefix(N))_T$(int_to_SI_prefix(T))_r$(int_to_SI_prefix(r))_omega$(omega)_alpha$(alpha).csv")
    save_Z_to_csv(Z, filename_Z)
    end

# Entry point of the script
isinteractive() || main(ARGS)
