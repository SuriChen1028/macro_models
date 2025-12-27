# hopenhayn.jl
# Continuous-Time Hopenhayn Model Implementation in Julia
# Based on Ben Moll's Hopenhayn Replication https://benjaminmoll.com/wp-content/uploads/2020/06/hopenhayn.m
using UnPack
using Parameters
using LinearAlgebra
using SparseArrays
using PATHSolver
using Plots

include("../src/finitediff.jl")

Base.@kwdef mutable struct ModelParameters
    # ones that need to be updated but are exogenous for firms 

    # structural parameters
    ρ::Float32=0.05
    ϵ::Float32=0.5
    φ::Float32=0.5
    α::Float32=0.5
    
    cf::Float32=0.05
    ce::Float32=0.6

    m = 1.0
    m̄ = 0.1
    η = 1000.0    
end

struct Grids
    nz::Int64
    z_grid_bar::Vector{Float64}
    z_grid::Vector{Float64}
    dz::Vector{Float64}
end

function initialize_grids(z_grid_max=1.0, z_grid_size=1000)

    z_grid_bar = range(0, z_grid_max; length=(2+z_grid_size))
    z_grid = z_grid_bar[2:end-1]
    dz = diff(z_grid_bar)

    return Grids(z_grid_size, z_grid_bar, z_grid, dz)
end

function P(Q, parameters::ModelParameters)
    @unpack ϵ = parameters
    return Q^(-ϵ)
end

function W(L, parameters::ModelParameters)
    @unpack φ = parameters
    return L^φ
end


function ψ(z, h_lb, h_ub)
    if z < h_lb || z > h_ub
        return 0.0
    else
        return 1.0 / (h_ub - h_lb)
    end
end

function π(z, p, w, parameters::ModelParameters)
    @unpack α, cf = parameters
    return (p * z)^(1 / (1 - α)) * w^(α / (α - 1)) * α^(α / (1 - α)) * (1 - α) - cf
end



function n(z, p, w, parameters::ModelParameters)
    @unpack α = parameters
    return (p * z * α / w)^(1 / (1 - α))
end

function y(z, p, w, parameters::ModelParameters)
    @unpack α = parameters
    return z * n(z, p, w, parameters)^α
end


function update_price(p_old, w_old, params::ModelParameters, grids::Grids;
    w_p=0.3, w_w=0.3)

    p = p_old
    w = w_old

    @unpack z_grid, z_grid_bar, dz, nz = grids
    @unpack ρ, ce, η, m̄ = params
    
    mu = - 0.01 * ones(nz)
    sigma = 0.01 * z_grid

    v_star(z, p, w) = 0.0
    v_star_vec = v_star.(z_grid, Ref(p), Ref(w))

    # solve for LCP
    bc = (Reflecting(), Reflecting())
    L_operator = min.(Diagonal(mu),0) * ∇1bcm(z_grid_bar, bc) + max.(Diagonal(mu),0) * ∇1bcp(z_grid_bar, bc) + 0.5 * Diagonal(sigma.^2) * ∇2bc(z_grid_bar, bc)
    A = I * ρ - L_operator
    q = - π.(z_grid, Ref(p), Ref(w), Ref(params)) .+ A * v_star_vec

    lb = zeros(nz)
    ub = 300*ones(nz) # Need to have upper bounds for the z.

    L = SparseMatrixCSC{Float64, Int32}(A)
    # code, z, w_val = PATHSolver.solve_mcp(L, q, lb, ub, lb)  # Solves (12)
    code, z, w_val = redirect_stdout(devnull) do
        PATHSolver.solve_mcp(L, q, lb, ub, lb)
    end

    # println("PATHSolver exited with code $code")

    # # Check convergence
    # if code != PATHSolver.MCP_Solved
    #     error("PATHSolver failed to converge. Exit code: $code")
    # end

    LCP_err = maximum(  (L * z + q) .*z )
    println("LCP error: $LCP_err")

    # # Check LCP error tolerance
    tol = 1e-6
    if LCP_err > tol
        error("LCP did not converge. Error: $LCP_err > tolerance: $tol")
    end

    v_vec = z .+ v_star_vec

    # Double check with mcpsolve
    # func(z) = L * z + q 
    # r = mcpsolve(func, lb, ub,
    #                 zeros(nz), # initial condition
    #                 reformulation = :smooth, # uses a so-called "Fischer function" to smooth out the problem
    #                 autodiff = :forward,
    #                 inplace = false,
    #                 ftol = 1e-12);
    
    # LCP_err = maximum( max.( (L * r.zero + q) .*r.zero  ))
    # println("LCP error: $LCP_err")  

    # Check LCP error tolerance
    # tol = 1e-6
    # if LCP_err > tol
    #     error("LCP did not converge. Error: $LCP_err > tolerance: $tol")
    # end

    # v_vec = r.zero .+ v_star_vec

    # find the exit cutoff 
    exit_index = findfirst(x -> x > v_star_vec[1], v_vec)
    println("Exit index: $exit_index")
    exit = zeros(nz)
    exit[1:exit_index-1] .= 1.0

    # compute the expected value ex ante 
    dz_mid = (dz[1:end-1] .+ dz[2:end]) ./ 2.0
    ψ_vec = ψ.(z_grid, 0.7, 1.0)
    EV = sum( v_vec .*  ψ_vec .* dz_mid ) - ce 

    println("Expected value EV: $EV")

    m = m̄ * exp( η * EV )
    println("Job creation rate m: $m")
    m = min(m, 1e4)
    # update the transition matrix according to exit rule 
    AA = copy(L_operator)
    for i in 1:exit_index-1
        AA[:, i] .= 0.0
        AA[i, i] = 1.0
    end

    # solve for g 
    println(size(AA), size(ψ_vec))
    g = - AA' \ ( m * ψ_vec)
    # println(g)
    g = max.(g, 0.0) # ensure non-negativity
    if all(g .== 0.0)
        println("Stationary distribution g is identically zero.")
        g = (1 .- exit ) ./ sum( (1 .- exit) .* dz_mid )  # uniform distribution over non-exiting states
    end
    # g = g ./ sum(g .* dz_mid) # normalize
    
    println(maximum(g),"\t", minimum(g))

    # ===================================================================
    # compute aggregates 
    # =================================================================
    n_vec = n.(z_grid, Ref(p), Ref(w), Ref(params))
    y_vec = y.(z_grid, Ref(p), Ref(w), Ref(params))

    println(maximum(n_vec))
    println(maximum(y_vec))

    N = max(min(sum( n_vec .* g .* dz_mid ), 1e3), 1e-10)
    Q = max( min(sum( y_vec .* g .* dz_mid ), 1e3), 1e-10)

    println("Aggregate N: $N, Q: $Q")

    # ===================================================================
    # update p

    p_implied = max(min(P(Q, params), 5.0), 0.01)
    err_p = abs(p_implied - p)
    p = w_p * p_implied + (1 - w_p) * p
    println("err_p: $err_p")
    # ===================================================================
    # update w 

    w_implied = max(min(W(N, params), 5.0), 0.01)
    err_w = abs(w_implied - w)
    w = w_w * w_implied + (1 - w_w) * w
    println("err_w: $err_w")

    return p, w, err_p, err_w, v_vec, g
end 


params = ModelParameters()
grids = initialize_grids()

# outer loop over p and w
max_iter_p = 1000
tol_p = 1e-8
max_iter_w = 100
tol_w = 1e-8

p_old = 0.7
w_old = 1.0

converged_p = false
converged_w = false

for iter_w in 1:max_iter_w
    println("Outer Iteration (w): $iter_w")

    global p_old, w_old, converged_p, converged_w
    w = copy(w_old)

    for iter_p in 1:max_iter_p
        println("  Inner Iteration (p): $iter_p")
        p = copy(p_old)

        println("  p_old: $p_old, w_old: $w_old")
        p_new, w_new, err_p, err_w, _, _ = update_price(p, w, params, grids, w_p=0.001, w_w=0.01)

        println("p iteration: $iter_p, err_p: $err_p")
        if err_p < tol_p || iter_p > max_iter_p
            if err_p < tol_p
                converged_p = true
            end
            break
        end
    
        p_old = p_new
    end

    p_new, w_new, err_p, err_w, _, _ = update_price(p_old, w, params, grids, w_p=0.01, w_w=0.1)

    w_old = w_new
    p_old = p_new
    
    if err_w < tol_w || iter_w > max_iter_w
        if err_w < tol_w
            converged_w = true
        end
        println("Converged after $iter_w iterations.")
        break
    end
end

if !converged_p
    println("Warning: Price p did not converge within the maximum number of iterations.")
end

if !converged_w
    println("Warning: Wage w did not converge within the maximum number of iterations.")
end


p_new, w_new, err_p, err_w, v_vec, g = update_price(p_old, w_old, params, grids, w_p=0.01, w_w=0.1)

plot(grids.z_grid, v_vec, title="Value Function v(z)", xlabel="z", ylabel="v(z)")
plot(grids.z_grid, g, title="Stationary Distribution g(z)", xlabel="z", ylabel="g(z)")