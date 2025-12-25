# functions.jl
using LinearAlgebra

abstract type BoundaryCondition end
struct Reflecting <: BoundaryCondition end
struct Absorbing <: BoundaryCondition end

abstract type FiniteDiffMethod end
struct backward1 <: FiniteDiffMethod end
struct forward1 <: FiniteDiffMethod end
struct central2 <: FiniteDiffMethod end


function greet_your_package_name()
    return "Hello FiniteDifferenceSC!"
end



function finitediff(x̄::AbstractVector, bc::Tuple{BoundaryCondition, BoundaryCondition},
    method::backward1)

    M = length(x̄) - 2  # number of interior points
    dx = diff(x̄)
    L = Tridiagonal(-1 ./ dx[2:M] ,  1 ./ dx[1:M] , zeros(M-1))

    # Apply boundary conditions
    if bc[1] isa Reflecting
        L[1, 1] +=  - 1 / dx[1]
    end

    if bc[1] isa Absorbing
        L[1, 1] = 1
    end

    if bc[2] isa Absorbing
        L[end, :] .= 0
        L[end, end] = 1
    end

    return L
end

function finitediff(x̄::AbstractVector, bc::Tuple{BoundaryCondition, BoundaryCondition},
    method::forward1)

    M = length(x̄) - 2  # number of interior points
    dx = diff(x̄)
    L = Tridiagonal(zeros(M-1) ,  - 1 ./ dx[1:M] , 1 ./ dx[2:M] )

    # Apply boundary conditions
    if bc[2] isa Reflecting
        L[end, end] +=  - 1 / dx[end]
    end

    if bc[1] isa Absorbing
        L[1, 1] = 1
    end

    if bc[2] isa Absorbing
        L[end, :] .= 0
        L[end, end] = 1
    end

    return L
end

function finitediff(x̄::AbstractArray, bc::Tuple{BoundaryCondition, BoundaryCondition}, 
    method::central2)

    M = length(x̄) - 2  # number of interior points
    dx = diff(x̄) # M+1 intervals
    dx_p = dx[2:end] # M elements
    dx_m = dx[1:end-1] # M elements
    dx_0 = (dx[2:end] .+ dx[1:end-1]) ./ 2.0

    L = Tridiagonal((1 ./dx_m .*  1 ./ dx_0)[2:M],  - ( 1 ./ dx_p .+ 1 ./ dx_m) .* 1 ./ dx_0 , (1 ./ dx_p .* 1 ./ dx_0)[1:M-1])

    # Apply boundary conditions
    if bc[1] isa Reflecting
        # L[1, 1] +=  1/ (dx_p[1] * dx_0[1])
        L[1,1] = - 1 / (dx_p[1] * dx_0[1])
    end

    if bc[2] isa Reflecting
        L[end, end] += 1 / (dx_p[end] * dx_0[end])
        # L[end, end-1] +=  - 1 / (dx_m[end] * dx_0[end])
    end

    if bc[1] isa Absorbing
        L[1, :] .= 0
        L[1, 1] = 1
    end

    if bc[2] isa Absorbing
        L[end, :] .= 0
        L[end, end] = 1
    end

    return L
end

∇1bcm(x̄::AbstractVector, bc::Tuple) = finitediff(x̄, bc, backward1())
∇1bcp(x̄::AbstractVector, bc::Tuple) = finitediff(x̄, bc, forward1())
∇2bc(x̄::AbstractVector, bc::Tuple) = finitediff(x̄, bc, central2())