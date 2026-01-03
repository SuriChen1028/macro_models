using LinearAlgebra

function extend_grid(grid::Vector{Float64})
    dgrid = diff(grid)
    extended_grid = [grid[1] - dgrid[1]; grid; grid[end] + dgrid[end]]
    return extended_grid
end