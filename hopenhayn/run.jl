include("../src/model.jl")

using Plots; gr()

params = ModelParameters()
grids = initialize_grids(params)

p = 2.0
v_init = zeros(length(grids.ϕ_grid))
@time v_bar = vfi(p, v_init, params, grids )

@time v_bar = vfi(p, v_init, params, grids )

p_min, p_max, p_size = 1.0, 2.0, 20
pvec = range(p_min, p_max, length=p_size)
entry_val = zeros(p_size)

entry_val .= [compute_net_entry_value(p, v_init, grids, params)[1] for p in pvec]


p1 = plot(pvec, entry_val, xlabel="Price p", ylabel="Net Entry Value", title="Net Entry Value vs Price", legend=false)
hline!([0.0], linestyle=:dash, color=:black)
display(p1)

@time p_star, v_bar, ϕ_star, ϕ_sample, s, m_star = compute_equilibrium(params, grids)

p2 = plot(grids.ϕ_grid, v_bar, xlabel="Productivity ϕ", ylabel="Value Function v(ϕ)", title="Equilibrium Value Function", legend=false)
display(p2)

println("\nResults:")
println("p_star = $p_star")
println("ϕ_star = $ϕ_star")
println("s = $s")
println("m_star = $m_star")

output_dist = q.(Ref(p_star), ϕ_sample, Ref(params))
histogram(log.(output_dist), normalize=:pdf)

# Pareto tail 

ϵ = 1.0e-10
xgrid = range(minimum(output_dist)+ϵ, stop=maximum(output_dist)-ϵ, length=100)
eccdf_vals = [mean(output_dist .> x) for x in xgrid]  # Counter CDF

p3 = plot(xgrid, eccdf_vals, 
    xscale=:log10, yscale=:log10,
    marker=:circle,
    xlabel="productivity", 
    ylabel="counter CDF",
    label="ECCDF",
    legend=:topright)
display(p3)

c_values = range(2.5, 5.0, length=10)

params_list = [ModelParameters(cf=c) for c in c_values]
p_star_list = [compute_equilibrium(p, grids)[1] for p in params_list]
