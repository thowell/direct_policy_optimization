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

function quadratic_cost(x,u,Q,R,x_ref,u_ref)
    (x-x_ref)'*Q*(x-x_ref) + (u-u_ref)'*R*(u-u_ref)
end

function stage_cost(model,x⁺,x,u,Q,R,x_ref,u_ref,h,c)
    # xm = xm_rk3_implicit(model,x⁺,x,u,h)
    #
    # ℓ1 = quadratic_cost(x,u,Q,R,x_ref,u_ref)
    # ℓ2 = quadratic_cost(xm,u,Q,R,x_ref,u_ref)
    # ℓ3 = quadratic_cost(x⁺,u,Q,R,x_ref,u_ref)
    #
    # return h[1]/6.0*ℓ1 + 4.0*h[1]/6.0*ℓ2 + h[1]/6.0*ℓ3 + c*h[1]
    ℓ1 = quadratic_cost(x,u,Q,R,x_ref,u_ref)
    return h[1]*ℓ1 + c*h[1]
end

function terminal_cost(x,Q,x_ref)
    return (x-x_ref)'*Q*(x-x_ref)
end

function objective(Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c

    s = 0
    for t = 1:T-2
        u = Z[idx.u[t][model.idx_u]]
        h = Z[idx.h[t]]

        x3 = Z[idx.x[t+2]]
        x3_ref = x_ref[t+2]
        Q3 = Q[t+2]

        s += h[1]*(x3-x3_ref)'*Q3*(x3-x3_ref)
        s += h[1]*(u-u_ref[t])'*R[t]*(u-u_ref[t])
        s += h[1]*c
    end

    return s
end

function objective_gradient!(∇l,Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R
    c = l.c

    for t = 1:T-2
        u = Z[idx.u[t][model.idx_u]]
        h = Z[idx.h[t]]

        x3 = Z[idx.x[t+2]]
        x3_ref = x_ref[t+2]
        Q3 = Q[t+2]

        ∇l[idx.x[t+2]] += 2.0*h[1]*Q3*(x3-x3_ref)
        ∇l[idx.u[t][model.idx_u]] += 2.0*h[1]*R[t]*(u-u_ref[t])
        ∇l[idx.h[t]] += c + (x3-x3_ref)'*Q3*(x3-x3_ref) + (u-u_ref[t])'*R[t]*(u-u_ref[t])
    end

    return nothing
end

objective(Z,prob::TrajectoryOptimizationProblem) = objective(Z,prob.obj,prob.model,prob.idx,prob.T)
objective_gradient!(∇l,Z,prob::TrajectoryOptimizationProblem) = objective_gradient!(∇l,Z,prob.obj,prob.model,prob.idx,prob.T)
