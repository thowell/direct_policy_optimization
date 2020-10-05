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
    x_ref
    u_ref
end

function quadratic_cost(x,u,Q,R,x_ref,u_ref)
    (x-x_ref)'*Q*(x-x_ref) + (u-u_ref)'*R*(u-u_ref)
end

function stage_cost(model,x⁺,x,u,Q,R,x_ref,u_ref)
    ℓ1 = quadratic_cost(x,u,Q,R,x_ref,u_ref)
    return ℓ1
end

function terminal_cost(x,Q,x_ref)
    return (x-x_ref)'*Q*(x-x_ref)
end

function objective(Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R

    s = 0
    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]
        x⁺ = Z[idx.x[t+1]]

        s += stage_cost(model,x⁺,x,u,Q[t],R[t],x_ref[t],u_ref[t])
    end
    x = view(Z,idx.x[T])
    s += terminal_cost(x,Q[T],x_ref[T])

    return s
end

function objective_gradient!(∇l,Z,l::QuadraticTrackingObjective,model,idx,T)
    x_ref = l.x_ref
    u_ref = l.u_ref
    Q = l.Q
    R = l.R

    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]
        x⁺ = Z[idx.x[t+1]]

        stage_cost_x(z) = stage_cost(model,x⁺,z,u,Q[t],R[t],x_ref[t],u_ref[t])
        stage_cost_u(z) = stage_cost(model,x⁺,x,z,Q[t],R[t],x_ref[t],u_ref[t])
        stage_cost_x⁺(z) = stage_cost(model,z,x,u,Q[t],R[t],x_ref[t],u_ref[t])

        ∇l[idx.x[t]] += ForwardDiff.gradient(stage_cost_x,x)
        ∇l[idx.u[t]] += ForwardDiff.gradient(stage_cost_u,u)
        ∇l[idx.x[t+1]] += ForwardDiff.gradient(stage_cost_x⁺,x⁺)
    end
    x = view(Z,idx.x[T])
    ∇l[idx.x[T]] += 2.0*Q[T]*(x-x_ref[T])

    return nothing
end

objective(Z,prob::TrajectoryOptimizationProblem) = objective(Z,prob.obj,prob.model,prob.idx,prob.T)
objective_gradient!(∇l,Z,prob::TrajectoryOptimizationProblem) = objective_gradient!(∇l,Z,prob.obj,prob.model,prob.idx,prob.T)

# Time penalty
mutable struct FreeTimeObjective{T} <: Objective
    c::T
end

function objective(Z,l::FreeTimeObjective,model,idx,T)
    J = 0
    for t = 1:T-1
        s = Z[idx.u[t][end]]
        J += s
    end
    return l.c*J
end

function objective_gradient!(∇l,Z,l::FreeTimeObjective,model,idx,T)
    for t = 1:T-1
        u = Z[idx.u[t][end]]
        ∇l[idx.u[t][end]] += l.c
    end
    return nothing
end


mutable struct FreeTimeTrackingObjective <: Objective
    track_obj::QuadraticTrackingObjective
    ft_obj::FreeTimeObjective
end

function objective(Z,obj::FreeTimeTrackingObjective,model,idx,T)
    x_ref = obj.track_obj.x_ref
    u_ref = obj.track_obj.u_ref
    Q = obj.track_obj.Q
    R = obj.track_obj.R
    c = obj.ft_obj.c

    s = 0
    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t][1:end-1]]
        h = Z[idx.u[t][end]]
        x⁺ = Z[idx.x[t+1]]

        s += h*(stage_cost(model,x⁺,x,u,Q[t],R[t][1:end-1,1:end-1],x_ref[t],u_ref[t][1:end-1]) + c)
    end
    x = view(Z,idx.x[T])
    s += terminal_cost(x,Q[T],x_ref[T])

end

function objective_gradient!(∇l,Z,obj::FreeTimeTrackingObjective,model,idx,T)
    x_ref = obj.track_obj.x_ref
    u_ref = obj.track_obj.u_ref
    Q = obj.track_obj.Q
    R = obj.track_obj.R
    c = obj.ft_obj.c

    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t][1:end-1]]
        h = Z[idx.u[t][end]]

        x⁺ = Z[idx.x[t+1]]

        stage_cost_x(z) = h*(stage_cost(model,x⁺,z,u,Q[t],R[t][1:end-1,1:end-1],x_ref[t],u_ref[t][1:end-1]) + c)
        stage_cost_u(z) = z[end]*(stage_cost(model,x⁺,x,z[1:end-1],Q[t],R[t][1:end-1,1:end-1],x_ref[t],u_ref[t][1:end-1]) + c)
        stage_cost_x⁺(z) = h*(stage_cost(model,z,x,u,Q[t],R[t][1:end-1,1:end-1],x_ref[t],u_ref[t][1:end-1]) + c)

        ∇l[idx.x[t]] += ForwardDiff.gradient(stage_cost_x,x)
        ∇l[idx.u[t]] += ForwardDiff.gradient(stage_cost_u,Z[idx.u[t]])
        ∇l[idx.x[t+1]] += ForwardDiff.gradient(stage_cost_x⁺,x⁺)
    end
    x = view(Z,idx.x[T])
    ∇l[idx.x[T]] += 2.0*Q[T]*(x-x_ref[T])
    return nothing
end
