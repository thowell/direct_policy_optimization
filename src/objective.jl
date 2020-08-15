# l(X,U) = (x-xT)'QT(x-xT) + h Σ {(x-xt)'Qt(x-xt) + (u-ut)'Rt(u-ut) + c}

abstract type Objective end

mutable struct MultiObjective <: Objective
    obj::Vector{Objective}
end

function objective(Z,obj::MultiObjective,model,idx,T)
    return sum([objective(Z,o,model,idx,T) for o in obj.obj])
end

function objective_gradient!(∇l,Z,obj::MultiObjective,model,idx,T)
    for o in obj.obj
        objective_gradient!(∇l,Z,o,model,idx,T)
    end
    return nothing
end

mutable struct QuadraticTrackingObjective <: Objective
    Q
    R
    c
    x_ref
    u_ref
end

function objective(Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c

    s = 0
    for t = 1:T
        x = Z[idx.x[t]]
        h = (t == 1 || t == 2) ? Z[idx.h[1]] : Z[idx.h[t-2]]
        s += h*(x-x_ref[t])'*Q[t]*(x-x_ref[t])

        t > T-2 && continue

        u = Z[idx.u[t][model.idx_u]]
        h = Z[idx.h[t]]

        s += h*(u-u_ref[t])'*R[t]*(u-u_ref[t])
        s += h*c
    end

    return s
end

function objective_gradient!(∇l,Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c

    for t = 1:T
        x = Z[idx.x[t]]
        h_idx = ((t == 1 || t == 2) ? idx.h[1] : idx.h[t-2])

        ∇l[idx.x[t]] += 2.0*Z[h_idx]*Q[t]*(x-x_ref[t])
        ∇l[h_idx] += (x-x_ref[t])'*Q[t]*(x-x_ref[t])

        t > T-2 && continue

        u = Z[idx.u[t][model.idx_u]]
        h = Z[idx.h[t]]

        ∇l[idx.u[t][model.idx_u]] += 2.0*h*R[t]*(u-u_ref[t])
        ∇l[idx.h[t]] += c + (u-u_ref[t])'*R[t]*(u-u_ref[t])
    end

    return nothing
end

mutable struct PenaltyObjective{T} <: Objective
    α::T
end

function objective(Z,l::PenaltyObjective,model,idx,T)
    J = 0
    for t = 1:T-2
        s = Z[idx.u[t][model.idx_s]]
        J += s
    end
    return l.α*J
end

function objective_gradient!(∇l,Z,l::PenaltyObjective,model,idx,T)
    for t = 1:T-2
        u = Z[idx.u[t][model.idx_s]]
        ∇l[idx.u[t][model.idx_s]] += l.α
    end
    return nothing
end

objective(Z,prob::TrajectoryOptimizationProblem) = objective(Z,prob.obj,prob.model,prob.idx,prob.T)
objective_gradient!(∇l,Z,prob::TrajectoryOptimizationProblem) = objective_gradient!(∇l,Z,prob.obj,prob.model,prob.idx,prob.T)
