function dynamics_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]
        x⁺ = Z[idx.x[t+1]]

        c[(t-1)*nx .+ (1:nx)] = (prob.free_time
            ? discrete_dynamics(model,x⁺,x,u,u[end],t)
            : discrete_dynamics(model,x⁺,x,u,prob.Δt,t))
    end

    return nothing
end

function dynamics_constraints_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    shift = 0

    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]
        x⁺ = Z[idx.x[t+1]]

        dyn_x(z) = (prob.free_time
            ? discrete_dynamics(model,x⁺,z,u,u[end],t)
            : discrete_dynamics(model,x⁺,z,u,prob.Δt,t))
        dyn_u(z) = (prob.free_time
            ? discrete_dynamics(model,x⁺,x,z,z[end],t)
            : discrete_dynamics(model,x⁺,x,z,prob.Δt,t)
            )
        dyn_x⁺(z) = (prob.free_time
            ? discrete_dynamics(model,z,x,u,u[end],t)
            : discrete_dynamics(model,z,x,u,prob.Δt,t)
            )

        r_idx = (t-1)*nx .+ (1:nx)

        ∇c[r_idx,idx.x[t]] = ForwardDiff.jacobian(dyn_x,x)
        ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        ∇c[r_idx,idx.x[t+1]] = ForwardDiff.jacobian(dyn_x⁺,x⁺)
    end

    return nothing
end

function sparse_dynamics_constraints_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    shift = 0

    for t = 1:T-1
        x = Z[idx.x[t]]
        u = Z[idx.u[t]]
        x⁺ = Z[idx.x[t+1]]

        dyn_x(z) = (prob.free_time
            ? discrete_dynamics(model,x⁺,z,u,u[end],t)
            : discrete_dynamics(model,x⁺,z,u,prob.Δt,t))
        dyn_u(z) = (prob.free_time
            ? discrete_dynamics(model,x⁺,x,z,z[end],t)
            : discrete_dynamics(model,x⁺,x,z,prob.Δt,t)
            )
        dyn_x⁺(z) = (prob.free_time
            ? discrete_dynamics(model,z,x,u,u[end],t)
            : discrete_dynamics(model,z,x,u,prob.Δt,t)
            )

        r_idx = (t-1)*nx .+ (1:nx)

        s = nx*nx
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x,x))
        shift += s

        # ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        s = nx*nu
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u,u))
        shift += s

        # ∇c[r_idx,idx.x[t+1]] .= ForwardDiff.jacobian(dyn_x,x)
        s = nx*nx
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x⁺,x⁺))
        shift += s
    end

    return nothing
end

function sparsity_dynamics_jacobian(prob::TrajectoryOptimizationProblem;
        r_shift=0)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    row = []
    col = []

    for t = 1:T-1
        r_idx = r_shift + (t-1)*nx .+ (1:nx)
        row_col!(row,col,r_idx,idx.x[t])
        row_col!(row,col,r_idx,idx.u[t])
        row_col!(row,col,r_idx,idx.x[t+1])
    end

    return collect(zip(row,col))
end
