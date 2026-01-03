# functions.jl
using LinearAlgebra
using SparseArrays

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



function finitediff(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple{BoundaryCondition, BoundaryCondition}, 
    method::backward1)
    
    M, N = [length(x̄[1])-2, length(x̄[2])-2]
    dx1 = diff(x̄[1])
    dx2 = diff(x̄[2])
    L = spzeros(M*N, M*N)

    A1 = finitediff(x̄[1], (nothing, nothing), method)
    A2 = finitediff(x̄[2], (nothing, nothing), method) 

    if bc[1] isa Reflecting
        A1[1,1] += -1 / dx1[1]
        A2[1,1] += -1 / dx2[1]

    end
    
    L = kron(I(N), A1) + kron(A2, I(M))

    if bc[1] isa Absorbing || bc[2] isa Absorbing
        
        boundary_idx = Int[]
        for j in 1:N
            for i in 1:M
                if i == 1 || i == M | j == 1 || j == N 
                    push!(boundary_idx, i + (j-1) * M)
                end
            end
        end

        L[boundary_idx, :] .= 0
        L[boundary_idx, boundary_idx] = I 
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
        L[end, end] += 1 / dx[end]
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

function finitediff(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple{BoundaryCondition, BoundaryCondition},
    method::forward1)

    M, N = [length(x̄[1])-2, length(x̄[2])-2]
    dx1 = diff(x̄[1])
    dx2 = diff(x̄[2])
    L = spzeros(M*N, M*N)

    A1 = finitediff(x̄[1], (nothing, nothing), method)
    A2 = finitediff(x̄[2], (nothing, nothing), method) 

    if bc[2] isa Reflecting
        A1[end,end] += 1 / dx1[end]
        A2[end,end] += 1 / dx2[end]
    end

    L = kron(I(N), A1) + kron(A2, I(M))    

    if bc[1] isa Absorbing || bc[2] isa Absorbing
        
        boundary_idx = Int[]
        for j in 1:N
            for i in 1:M
                if i == 1 || i == M || j == 1 || j == N 
                    push!(boundary_idx, i + (j-1) * M)
                end
            end
        end

        L[boundary_idx, :] .= 0
        L[boundary_idx, boundary_idx] = I 
    end

    return L
end

function finitediff(x̄::AbstractVector, bc::Tuple{BoundaryCondition, BoundaryCondition}, 
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

function finitediff(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple{BoundaryCondition, BoundaryCondition}, 
    method::central2)

    M, N = [length(x̄[1]) - 2, length(x̄[2]) - 2]
    dx1 = diff(x̄[1])
    dx2 = diff(x̄[2]) 
    dx1_p = dx1[2:end] # M elements
    dx1_m = dx1[1:end-1] # M elements
    dx1_0 = (dx1[2:end] .+ dx1[1:end-1]) ./ 2.0
    dx2_p = dx2[2:end] # N elements
    dx2_m = dx2[1:end-1] # N elements
    dx2_0 = (dx2[2:end] .+ dx2[1:end-1]) ./ 2.0
    L = spzeros(M*N, M*N)

    A1 = finitediff(x̄[1], (nothing, nothing), method)
    A2 = finitediff(x̄[2], (nothing, nothing), method)
    
    if bc[1] isa Reflecting
        A1[1,1] = - 1 / (dx1_p[1] * dx1_0[1])
        A2[1,1] = - 1 / (dx2_p[1] * dx2_0[1])
    end

    if bc[2] isa Reflecting
        A1[end, end] += 1 / (dx1_p[end] * dx1_0[end])
        A2[end, end] += 1 / (dx2_p[end] * dx2_0[end])
    end

    L = kron(I(N), A1) + kron(A2, I(M))

    if bc[1] isa Absorbing || bc[2] isa Absorbing
        
        boundary_idx = Int[]
        for j in 1:N
            for i in 1:M
                if i == 1 || i == M || j == 1 || j == N 
                    push!(boundary_idx, i + (j-1) * M)
                end
            end
        end

        L[boundary_idx, :] .= 0
        L[boundary_idx, boundary_idx] = I
    end
    return L

end

∇1bcm(x̄::AbstractVector, bc::Tuple) = finitediff(x̄, bc, backward1())
∇1bcm(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple) = finitediff(x̄, bc, backward1())
∇1bcp(x̄::AbstractVector, bc::Tuple) = finitediff(x̄, bc, forward1())
∇1bcp(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple) = finitediff(x̄, bc, forward1())
∇2bc(x̄::AbstractVector, bc::Tuple) = finitediff(x̄, bc, central2())
∇2bc(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple) = finitediff(x̄, bc, central2())

# Methods that handle nothing boundary conditions - returns operators without BC modifications
finitediff(x̄::AbstractVector, bc::Tuple{Nothing, Nothing}, method::backward1) = finitediff_no_bc(x̄, method)
finitediff(x̄::AbstractVector, bc::Tuple{Nothing, Nothing}, method::forward1) = finitediff_no_bc(x̄, method)
finitediff(x̄::AbstractVector, bc::Tuple{Nothing, Nothing}, method::central2) = finitediff_no_bc(x̄, method)
finitediff(x̄::Tuple{AbstractVector, AbstractVector}, bc::Tuple{Nothing, Nothing}, method::FiniteDiffMethod) = finitediff_no_bc(x̄, method)

∇1m(x̄::AbstractVector) = finitediff_no_bc(x̄, backward1())
∇1p(x̄::AbstractVector) = finitediff_no_bc(x̄, forward1())
∇2(x̄::AbstractVector) = finitediff_no_bc(x̄, central2())
# Helper function that creates operators without boundary condition modifications
function finitediff_no_bc(x̄::AbstractVector, method::backward1)
    M = length(x̄) - 2
    dx = diff(x̄)
    return Tridiagonal(-1 ./ dx[2:M], 1 ./ dx[1:M], zeros(M-1))
end

function finitediff_no_bc(x̄::AbstractVector, method::forward1)
    M = length(x̄) - 2
    dx = diff(x̄)
    return Tridiagonal(zeros(M-1), -1 ./ dx[2:M+1], 1 ./ dx[2:M])
end

function finitediff_no_bc(x̄::AbstractVector, method::central2)
    M = length(x̄) - 2
    dx = diff(x̄)
    dx_p = dx[2:end]
    dx_m = dx[1:end-1]
    dx_0 = (dx[2:end] .+ dx[1:end-1]) ./ 2.0
    return Tridiagonal((1 ./dx_m .* 1 ./ dx_0)[2:M], -(1 ./ dx_p .+ 1 ./ dx_m) .* 1 ./ dx_0, (1 ./ dx_p .* 1 ./ dx_0)[1:M-1])
end

function finitediff_no_bc(x̄::Tuple{AbstractVector, AbstractVector}, method::FiniteDiffMethod)
    M, N = length(x̄[1]) - 2, length(x̄[2]) - 2
    Lx = finitediff_no_bc(x̄[1], method)
    Ly = finitediff_no_bc(x̄[2], method)
    return kron(sparse(I, N, N), Lx) + kron(Ly, sparse(I, M, M))
end