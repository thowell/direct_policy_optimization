include("biped/f.jl")
include("biped/g.jl")
include("biped/utils.jl")

mutable struct Biped{T}
    l1::T
    l2::T
    Tm::Int

    nx::Int
    nu::Int
    nw::Int
end

function dynamics(model::Biped,x,u,w)
    f(x) + g(x)*u
end

nx = 10
nu = 4
nw = 0
model = Biped(0.2755,0.288,0,nx,nu,nw)

mutable struct PenaltyObjective{T} <: Objective
    α::T
    pfz_des
    idx_t
end

function objective(Z,l::PenaltyObjective,model::Biped,idx,T)
    J = 0
    for t in l.idx_t
        q = Z[idx.x[t][1:5]]
        # pfx = kinematics(model,q)[1]
        pfz = kinematics(model,q)[2]
        # J += (pfx - l.pfz_des[1][t])*(pfx - l.pfz_des[1][t])
        J += (pfz - l.pfz_des[t])*(pfz - l.pfz_des[t])
    end
    return l.α*J
end

function objective_gradient!(∇l,Z,l::PenaltyObjective,model::Biped,idx,T)
    for t in l.idx_t
        q = Z[idx.x[t][1:5]]
        # tmpx(w) = kinematics(model,w)[1]
        tmpz(w) = kinematics(model,w)[2]
        # pfx = tmpx(q)
        pfz = tmpz(q)
        # ∇l[idx.x[t][1:5]] += 2.0*l.α*(pfx - l.pfz_des[1][t])*ForwardDiff.gradient(tmpx,q)
        ∇l[idx.x[t][1:5]] += 2.0*l.α*(pfz - l.pfz_des[t])*ForwardDiff.gradient(tmpz,q)
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
