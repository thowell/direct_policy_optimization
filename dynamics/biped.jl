include("biped/f.jl")
include("biped/g.jl")
include("biped/utils.jl")

mutable struct Biped{T}
    l1::T
    l2::T
    Tm::Int
    nx::Int
    nu::Int
end

function dynamics(model::Biped,x,u)
    f(x) + g(x)*u
end

nx = 10
nu = 4
model = Biped(0.2755,0.288,0,nx,nu)

mutable struct PenaltyObjective{T} <: Objective
    α::T
    pfz_des::T
    idx_t
end

function objective(Z,l::PenaltyObjective,model::Biped,idx,T)
    J = 0
    for t in l.idx_t
        q = Z[idx.x[t][1:5]]
        pfz = kinematics(model,q)[2]
        J += (pfz - l.pfz_des)*(pfz - l.pfz_des)
    end
    return l.α*J
end

function objective_gradient!(∇l,Z,l::PenaltyObjective,model::Biped,idx,T)
    for t in l.idx_t
        q = Z[idx.x[t][1:5]]
        tmp(w) = kinematics(model,w)[2]
        pfz = tmp(q)
        ∇l[idx.x[t][1:5]] += 2.0*l.α*(pfz - l.pfz_des)*ForwardDiff.gradient(tmp,q)
    end
    return nothing
end

# prob.obj.obj[2].α=3.7
# obj_tmp(z) = objective(z,prob.obj.obj[2],model,prob.idx,T)
# ForwardDiff.gradient(obj_tmp,Z0)
#
# ∇obj_ = zero(Z0)
# objective_gradient!(∇obj_,Z0,prob.obj.obj[2],model,prob.idx,T)
# norm(∇obj_ - ForwardDiff.gradient(obj_tmp,Z0))
