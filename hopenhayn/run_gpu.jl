include("src/model_gpu.jl")

using Plots; gr()
using CUDA

# Check if GPU is available
println("CUDA available: ", CUDA.functional())
println("GPU device: ", CUDA.device())

params = ModelParameters()

println("\nInitializing grids with increased sizes...")
println("Grid size: 1000, Draw sizes: 5000 each")
@time grids = initialize_grids(params)

println("\nTesting VFI...")
p = 2.0
v_init = CUDA.zeros(length(grids.ϕ_grid))
@time v_bar = vfi(p, v_init, params, grids)

println("\nComputing entry values across price range...")
p_min, p_max, p_size = 1.0, 2.0, 20
pvec = range(p_min, p_max, length=p_size)
entry_val = zeros(p_size)

@time entry_val .= [compute_net_entry_value(p, v_init, grids, params)[1] for p in pvec]

p1 = plot(pvec, entry_val, xlabel="Price p", ylabel="Net Entry Value", 
          title="Net Entry Value vs Price (GPU, Grid=1000)", legend=false)
hline!([0.0], linestyle=:dash, color=:black)
display(p1)

println("\nComputing full equilibrium...")
@time p_star, v_bar, ϕ_star, ϕ_sample, s, m_star = compute_equilibrium(params, grids)

p2 = plot(Array(grids.ϕ_grid), v_bar, xlabel="Productivity ϕ", 
          ylabel="Value Function v(ϕ)", 
          title="Equilibrium Value Function (GPU)", legend=false)
display(p2)

println("\nResults:")
println("p_star = $p_star")
println("ϕ_star = $ϕ_star")
println("s = $s")
println("m_star = $m_star")

println("\nComputing output distribution...")
output_dist = q.(Ref(p_star), ϕ_sample, Ref(params))

p3 = histogram(log.(output_dist), normalize=:pdf, 
               xlabel="Log Output", ylabel="Density",
               title="Output Distribution (5M firms)", legend=false)
display(p3)

# ECCDF plot
println("\nComputing ECCDF...")
ϵ = 1.0e-10
xgrid = range(minimum(output_dist)+ϵ, stop=maximum(output_dist)-ϵ, length=100)
eccdf_vals = 1 .- [mean(output_dist .<= x) for x in xgrid]

p4 = plot(xgrid, eccdf_vals, 
    xscale=:log10, yscale=:log10,
    marker=:circle,
    xlabel="Output", 
    ylabel="Counter CDF",
    title="ECCDF (5M firms, GPU)",
    label="ECCDF",
    legend=:topright)
display(p4)

println("\nAll computations complete!")
