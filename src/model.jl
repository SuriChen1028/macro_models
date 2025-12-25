# model.jl
# Julia translation of QuantEcon code for Hopenhayn (1992) model
# Original Python code by Thomas J. Sargent and John Stachurski
using Distributions
using Random
using Interpolations
using UnPack

Base.@kwdef mutable struct ModelParameters
    β::Float32=0.95
    θ::Float32=0.3
    cf::Float32=4.0
    ce::Float32=1.0
    w::Float32=1.0
    m_a::Float32=-0.012
    σ_a::Float32=0.1
    m_e::Float32=1.0
    σ_e::Float32=0.2
    
end

struct Grids
    ϕ_grid::Vector{Float64}
    A_draws::Vector{Float64}
    E_draws::Vector{Float64}
end

function initialize_grids(modelparams::ModelParameters;
    ϕ_grid_max=5.0,
    ϕ_grid_size=100,
    E_draws_size=200,
    A_draws_size=200,
    seednum=123)


    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = modelparams

    @assert m_a + σ_a^2 /(2.0 * (1 - θ)) < 0

    ϕ_grid = range(0, ϕ_grid_max; length=ϕ_grid_size)

    Random.seed!(seednum)

    d = Normal()
    A_draws = exp.(m_a .+ σ_a .* rand(d, A_draws_size) )
    E_draws = exp.(m_e .+ σ_e .* rand(d, E_draws_size) )

    return Grids(ϕ_grid, A_draws, E_draws)
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
    ϕ_bar, ϕ_vec, seednum, 
    parameters::ModelParameters, num_firms
    )

    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters

    Random.seed!(seednum)
    Z = rand(Normal(), (2, num_firms))
    # incumbent
    incumbent_draws = ϕ_vec .* exp.(m_a .+ σ_a .* Z[1, :] )
    new_firm_draws = exp.(m_e .+ σ_e .* Z[2, :] )
    return ifelse.(ϕ_vec .>= ϕ_bar, incumbent_draws, new_firm_draws)
end

function simulate_firms(
    ϕ_bar, parameters::ModelParameters;
    sim_length=200, num_firms=1_000_000, seednum=12
)
    ϕ_vec = ones((num_firms, )) .* ϕ_bar 
    for t in 1:sim_length
        ϕ_vec = update_cross_section(ϕ_bar, ϕ_vec, seednum + t, parameters, num_firms)
    end
    return ϕ_vec
end

function _compute_exp_valute_at_phi(v, ϕ, grids::Grids)
    ϕ_grid, A_draws, _ = grids.ϕ_grid, grids.A_draws, grids.E_draws
    Aϕ = A_draws .* ϕ
    itp = LinearInterpolation(ϕ_grid, v, extrapolation_bc=Flat())
    vAϕ = itp.(Aϕ)
    return mean(vAϕ)
end


function compute_exp_value(v, grids::Grids)
    return map(ϕ -> _compute_exp_valute_at_phi(v, ϕ, grids), grids.ϕ_grid)
end

function T(v, p, grids::Grids, parameters::ModelParameters)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters
    EvAϕ = compute_exp_value(v, grids)
    return π.(Ref(p), grids.ϕ_grid, Ref(parameters)) .+ β .* max.(0.0, EvAϕ)
end

function get_threshold(v, grids::Grids)

    ϕ_grid, _, _ = grids.ϕ_grid, grids.A_draws, grids.E_draws
    EvAϕ = compute_exp_value(v, grids)
    i = searchsortedfirst(EvAϕ, 0.0)
    return ϕ_grid[i]
end

function vfi(p, v_init, parameters::ModelParameters, grids::Grids; 
    max_iter=1000, tol=1e-8)

    function cond_function(state)
        i, v, error = state
        return (error > tol) && (i < max_iter)
    end

    function body_function(state)
        i, v, error = state
        v_new = T(v, p, grids, parameters)
        error = maximum(abs.(v_new .- v))
        return (i + 1, v_new, error)
    end

    init_state = (1, v_init, tol+1.0)
    state = init_state
    while cond_function(state)
        state = body_function(state)
    end

    i, v, error = state
    println("Converged in $i iterations with error $error")
    return v
end

function compute_net_entry_value(p, v_init, grids::Grids, parameters::ModelParameters)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters
    ϕ_grid, _, E_draws = grids.ϕ_grid, grids.A_draws, grids.E_draws
    v_bar = vfi(p, v_init, parameters, grids)
    entry_itp = LinearInterpolation(ϕ_grid, v_bar, extrapolation_bc=Flat())
    v_ϕ = entry_itp.(E_draws)

    return mean(v_ϕ) - ce, v_bar
end

function compute_p_star(parameters::ModelParameters, grids::Grids; p_min = 1.0, p_max = 2.0, tol = 1.0e-8)
    @unpack β, θ, cf, ce, w, m_a, σ_a, m_e, σ_e = parameters
    lower, upper = p_min, p_max
    ϕ_grid, _, _ = grids.ϕ_grid, grids.A_draws, grids.E_draws
    v_init = zeros(length(ϕ_grid))
    
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
    p_star = compute_p_star(parameters, grids)
    v_bar = vfi(p_star, zeros(length(grids.ϕ_grid)), parameters, grids)
    ϕ_star = get_threshold(v_bar, grids)

    ϕ_sample = simulate_firms(ϕ_star, parameters)

    demand = 1. ./ p_star 
    pre_normalized_supply = mean(q.(Ref(p_star), ϕ_sample, Ref(parameters)) )
    s = demand / pre_normalized_supply
    m_star = s * mean( ϕ_sample .< ϕ_star )

    return p_star, v_bar, ϕ_star, ϕ_sample, s, m_star
end

function ecdf(x, data)
    return mean(data .> x)
end