export SquareCSop, RombergOP, forward_model

import Base.*
import Base.-
import Base.adjoint
using Flux: cpu, gpu

mutable struct SquareCSop
    A::AbstractArray{Float32,2}
    nx::Int64
    ny::Int64
    nc::Int64
    s::Float32
end

function SquareCSop(nc::Int; s::Float32=3f0)

    A = randn(Float32, nc, nc) + s*I
    A = A/opnorm(A)

    return SquareCSop(A, 1, 1, nc, s)
end


function SquareCSop(A::AbstractArray{Float32,2}, nc::Int; s::Float32=3f0)
    return SquareCSop(A, 1, 1, nc, s)
end


function adjoint(A::SquareCSop)
    return SquareCSop(adjoint(A.A), A.nc; s=A.s)
end


function -(A::SquareCSop)
    return SquareCSop(-A.A, A.nc; s=A.s)
end


struct RombergOP
    A::joLinearOperator{Float32,Float32}
    nx::Int64
    ny::Int64
    nc::Int64
    sr::Float32
end


function RombergOP(nx::Int, ny::Int, nc::Int, sr::Float32)

    M = joRomberg(prod([nx, ny]); DDT=Float32, RDT=Float32)
    R = joRestriction(
        prod([nx, ny]), randperm(prod([nx, ny]))[1:Int(cld(prod([nx, ny]), sr))];
        DDT=Float32, RDT=Float32
    )

    A_flat = R*M
    A = A_flat'*A_flat

    return RombergOP(A, nx, ny, nc, sr)
end


function RombergOP(nc::Int, sr::Float32)

    M = joRomberg(nc; DDT=Float32, RDT=Float32)
    R = joRestriction(
        nc, randperm(nc)[1:Int(cld(nc, sr))];
        DDT=Float32, RDT=Float32
    )

    A_flat = R*M
    A = A_flat'*A_flat

    return RombergOP(A, 1, 1, nc, sr)
end


function adjoint(A::RombergOP)
    return RombergOP(A.A', A.ny, A.nx, A.nc, A.sr)
end

function -(A::RombergOP)
    return RombergOP(-A.A, A.nx, A.ny, A.nc, A.sr)
end


function *(CS::SquareCSop , X::AbstractArray{Float32,4})
    nb = size(X, 4)
    return reshape(CS.A*reshape(X, :, nb), CS.nx, CS.ny, CS.nc, :)
end


function *(CS::RombergOP , X::Array{Float32,4})
    nb = size(X, 4)
    return reshape(CS.A*reshape(X, :, nb), CS.nx, CS.ny, CS.nc, :)
end


function forward_model(X::Array{Float32,4})

    nx, ny, nc, nb = size(X)

    M = joRomberg(prod([nx, ny]); DDT=Float32, RDT=Float32)
    R = joRestriction(
        prod([nx, ny]), randperm(prod([nx, ny]))[1:cld(prod([nx, ny]), 2)];
        DDT=Float32, RDT=Float32
    )

    A_flat = R*M
    A = A_flat'*A_flat

    return reshape(A*reshape(X, :, nb), nx, ny, nc, nb)
end
