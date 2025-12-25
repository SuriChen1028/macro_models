
using Distributions
using Random
using Interpolations
using UnPack
using CUDA
using Statistics

# Include GPU interpolation
include("gpu_interp.jl")

Base.@kwdef struct ModelParameters  # Made immutable (removed 'mutable')
    β::Float32=0.95f0
    θ::Float32=0.3f0
    cf::Float32=4.0f0
    ce::Float32=1.0f0
    w::Float32=1.0f0
    m_a::Float32=-0.012f0
    σ_a::Float32=0.1f0
    m_e::Float32=1.0f0
    σ_e::Float32=0.2f0
end

struct Grids
    ϕ_grid::CuVector{Float64}
    A_draws::CuVector{Float64}
    E_draws::CuVector{Float64}
end

function initialize_grids(modelparams::ModelParameters;
    ϕ_grid_max=5.0,
    ϕ_grid_size=1000,        # Increased from 100
    E_draws_size=5000,        # Increased from 200
    A_draws_size=5000,        # Increased from 200
    seednum=1234)

    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = modelparams

    @assert m_a + σ_a^2 /(2.0 * (1 - θ)) < 0

    ϕ_grid = range(0, ϕ_grid_max; length=ϕ_grid_size)

    Random.seed!(seednum)

    d = Normal()
    A_draws = exp.(m_a .+ σ_a .* rand(d, A_draws_size))
    E_draws = exp.(m_e .+ σ_e .* rand(d, E_draws_size))

    # Convert to GPU arrays
    return Grids(CuArray(collect(ϕ_grid)), CuArray(A_draws), CuArray(E_draws))
end

function π(p, ϕ, modelparams::ModelParameters)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = modelparams
    return θ^(θ/(1 - θ)) * (1 - θ) * (p * ϕ)^(1/(1 - θ)) * w^(θ/(θ-1)) - cf 
end

function q(p, ϕ, modelparams::ModelParameters)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = modelparams
    return (θ * p * ϕ / w)^(θ / (1- θ)) * ϕ
end

function update_cross_section(
    ϕ_bar, ϕ_vec::CuVector, seednum, 
    parameters::ModelParameters, num_firms
    )

    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters

    Random.seed!(seednum)
    # Generate random numbers on GPU
    Z = CUDA.randn(Float64, (2, num_firms))
    
    # All operations on GPU
    incumbent_draws = ϕ_vec .* exp.(m_a .+ σ_a .* Z[1, :])
    new_firm_draws = exp.(m_e .+ σ_e .* Z[2, :])
    return ifelse.(ϕ_vec .>= ϕ_bar, incumbent_draws, new_firm_draws)
end

function simulate_firms(
    ϕ_bar, parameters::ModelParameters;
    sim_length=200, num_firms=5_000_000, seednum=12  # Increased from 1M
)
    ϕ_vec = CUDA.ones(num_firms) .* ϕ_bar 
    for t in 1:sim_length
        ϕ_vec = update_cross_section(ϕ_bar, ϕ_vec, seednum + t, parameters, num_firms)
    end
    return ϕ_vec
end

function _compute_exp_valute_at_phi(v, ϕ, grids::Grids)
    # Move to CPU for interpolation (interpolation not GPU-optimized)
    ϕ_grid_cpu = Array(grids.ϕ_grid)
    A_draws_cpu = Array(grids.A_draws)
    v_cpu = Array(v)
    
    Aϕ = A_draws_cpu .* ϕ
    itp = LinearInterpolation(ϕ_grid_cpu, v_cpu, extrapolation_bc=Flat())
    vAϕ = itp.(Aϕ)
    return mean(vAϕ)
end

function compute_exp_value(v, grids::Grids)
    # Vectorized GPU computation - no loops!
    n_grid = length(grids.ϕ_grid)
    n_draws = length(grids.A_draws)
    
    # Create a matrix: each row is A_draws * ϕ_grid[i] really? it's not each column?
    # Shape: (n_draws, n_grid)
    ϕ_grid_reshaped = reshape(grids.ϕ_grid, 1, n_grid)
    A_draws_reshaped = reshape(grids.A_draws, n_draws, 1)
    Aϕ_matrix = A_draws_reshaped .* ϕ_grid_reshaped  # Broadcasting on GPU
    
    # Flatten to interpolate all at once
    Aϕ_flat = vec(Aϕ_matrix)
    
    # Single GPU interpolation for all points
    vAϕ_flat = gpu_linear_interp(grids.ϕ_grid, v, Aϕ_flat)
    
    # Reshape back to (n_draws, n_grid)
    vAϕ_matrix = reshape(vAϕ_flat, n_draws, n_grid)
    
    # Mean across draws (first dimension) for each grid point
    result = vec(mean(vAϕ_matrix, dims=1))
    
    return result
end

function T(v, p, grids::Grids, parameters::ModelParameters)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters
    EvAϕ = compute_exp_value(v, grids)
    # Broadcast operations on GPU
    return π.(Ref(p), grids.ϕ_grid, Ref(parameters)) .+ β .* max.(0.0, EvAϕ)
end

function get_threshold(v, grids::Grids)
    EvAϕ = compute_exp_value(v, grids)
    EvAϕ_cpu = Array(EvAϕ)
    ϕ_grid_cpu = Array(grids.ϕ_grid)
    i = searchsortedfirst(EvAϕ_cpu, 0.0)
    return ϕ_grid_cpu[i]
end

function vfi(p, v_init, parameters::ModelParameters, grids::Grids; 
    max_iter=1000, tol=1e-8)

    v = CuArray(v_init)
    
    for i in 1:max_iter
        v_new = T(v, p, grids, parameters)
        error = maximum(abs.(v_new .- v))  # GPU reduction
        
        if error < tol
            println("Converged in $i iterations with error $error")
            return v_new
        end
        v = v_new
    end
    
    println("Reached max iterations ($max_iter)")
    return v
end

function compute_net_entry_value(p, v_init, grids::Grids, parameters::ModelParameters)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters
    
    v_bar = vfi(p, v_init, parameters, grids)
    
    # GPU interpolation - stays on device!
    v_ϕ = gpu_linear_interp(grids.ϕ_grid, v_bar, grids.E_draws)

    return mean(v_ϕ) - ce, v_bar
end

function compute_p_star(parameters::ModelParameters, grids::Grids; p_min = 1.0, p_max = 2.0, tol = 1.0e-8)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters
    lower, upper = p_min, p_max
    v_init = CUDA.zeros(length(grids.ϕ_grid))
    
    while upper - lower > tol
        mid = (upper + lower) / 2.0
        entry_val, v_bar = compute_net_entry_value(mid, v_init, grids, parameters)
        if entry_val > 0.0
            upper = mid
        else
            lower = mid
        end
    end
    return (upper + lower) / 2.0   
end

function compute_equilibrium(parameters::ModelParameters, grids::Grids)
    println("Computing equilibrium price...")
    p_star = compute_p_star(parameters, grids)
    
    println("Computing value function at equilibrium price...")
    v_bar = vfi(p_star, CUDA.zeros(length(grids.ϕ_grid)), parameters, grids)
    
    println("Computing exit threshold...")
    ϕ_star = get_threshold(v_bar, grids)

    println("Simulating firms...")
    ϕ_sample = simulate_firms(ϕ_star, parameters)

    # Move to CPU for final calculations
    ϕ_sample_cpu = Array(ϕ_sample)
    
    demand = 1.0 / p_star 
    pre_normalized_supply = mean(q.(Ref(p_star), ϕ_sample_cpu, Ref(parameters)))
    s = demand / pre_normalized_supply
    m_star = s * mean(ϕ_sample_cpu .>= ϕ_star)

    return p_star, Array(v_bar), ϕ_star, ϕ_sample_cpu, s, m_star
end

function ecdf(x, data)
    return mean(data .> x)
end
