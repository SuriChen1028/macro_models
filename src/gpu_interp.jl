using CUDA
using Interpolations

"""
GPU Linear Interpolation Module

This implements linear interpolation entirely on the GPU.
"""

# Binary search on GPU to find the bracketing indices
function binary_search_gpu(x_grid, x_query, n)
    """Find index i such that x_grid[i] <= x_query < x_grid[i+1]"""
    
    # Handle edge cases
    if x_query <= x_grid[1]
        return 1
    end
    if x_query >= x_grid[n]
        return n - 1
    end
    
    # Binary search
    left = 1
    right = n
    
    while right - left > 1
        mid = (left + right) ÷ 2 
        if x_query < x_grid[mid]
            right = mid
        else
            left = mid
        end
    end
    
    return left
end

# GPU kernel for linear interpolation
function linear_interp_kernel!(output, x_grid, y_values, x_query, n_grid, n_query)
    # Get thread index
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Check bounds
    if idx <= n_query
        x = x_query[idx]
        
        # Handle FLAT extrapolation at bounds FIRST (before binary search)
        if x <= x_grid[1]
            output[idx] = y_values[1]
        elseif x >= x_grid[n_grid]
            output[idx] = y_values[n_grid]
        else
            # Find bracketing indices using binary search (only for interior points)
            i = binary_search_gpu(x_grid, x, n_grid)
            
            # Linear interpolation
            x0 = x_grid[i]
            x1 = x_grid[i + 1]
            y0 = y_values[i]
            y1 = y_values[i + 1]
            
            # Interpolate: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            t = (x - x0) / (x1 - x0)
            output[idx] = y0 + (y1 - y0) * t
        end
    end
    
    return nothing
end

"""
    gpu_linear_interp(x_grid::CuVector, y_values::CuVector, x_query::CuVector)

Perform linear interpolation on GPU.

# Arguments
- `x_grid`: Sorted vector of x coordinates (on GPU)
- `y_values`: Corresponding y values (on GPU)
- `x_query`: Points at which to interpolate (on GPU)

# Returns
- Interpolated values at x_query points (on GPU)
"""
function gpu_linear_interp(x_grid::CuVector, y_values::CuVector, x_query::CuVector)
    n_grid = length(x_grid)
    n_query = length(x_query)
    
    # Allocate output on GPU
    output = CUDA.zeros(Float64, n_query)
    
    # Launch kernel
    threads_per_block = 256
    num_blocks = cld(n_query, threads_per_block)  # Ceiling division
    
    @cuda threads=threads_per_block blocks=num_blocks linear_interp_kernel!(
        output, x_grid, y_values, x_query, n_grid, n_query
    )
    
    return output
end

"""
    gpu_linear_interp(x_grid::Vector, y_values::Vector, x_query::Vector)

CPU wrapper that handles GPU transfers automatically.
"""
function gpu_linear_interp(x_grid::Vector, y_values::Vector, x_query::Vector)
    # Transfer to GPU
    x_grid_gpu = CuArray(x_grid)
    y_values_gpu = CuArray(y_values)
    x_query_gpu = CuArray(x_query)
    
    # Interpolate on GPU
    result_gpu = gpu_linear_interp(x_grid_gpu, y_values_gpu, x_query_gpu)
    
    # Transfer back to CPU
    return Array(result_gpu)
end

# Test function
function test_gpu_interp()
    println("Testing GPU Linear Interpolation...")
    
    # Create test data
    x_grid = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
    y_values = Float64[1.0, 4.0, 9.0, 16.0, 25.0]  # y = x^2
    
    # Query points
    x_query = Float64[1.5, 2.5, 3.5, 4.5]
    
    println("\nInput:")
    println("x_grid: ", x_grid)
    println("y_values: ", y_values)
    println("x_query: ", x_query)
    
    # Interpolate on GPU
    result = gpu_linear_interp(x_grid, y_values, x_query)
    
    println("\nGPU Result:")
    println("Interpolated values: ", result)
    
    # Compare with Interpolations.jl
    itp = LinearInterpolation(x_grid, y_values)
    cpu_result = itp.(x_query)
    
    println("\nCPU (Interpolations.jl) Result:")
    println("Interpolated values: ", cpu_result)
    
    println("\nDifference:")
    println("Max error: ", maximum(abs.(result .- cpu_result)))
    
    if maximum(abs.(result .- cpu_result)) < 1e-10
        println("\n✓ Test PASSED - GPU interpolation matches CPU!")
    else
        println("\n✗ Test FAILED - Results don't match")
    end
end
