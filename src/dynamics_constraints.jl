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
        h = Z[idx.h[t]]
        x⁺ = Z[idx.x[t+1]]

        c[(t-1)*nx .+ (1:nx)] = discrete_dynamics(model,x⁺,x,u,h,t)

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            c[nx*(T-1) + t] = h⁺ - h
        end
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
        h = Z[idx.h[t]]
        x⁺ = Z[idx.x[t+1]]

        dyn_x(z) = discrete_dynamics(model,x⁺,z,u,h,t)
        dyn_u(z) = discrete_dynamics(model,x⁺,x,z,h,t)
        dyn_h(z) = discrete_dynamics(model,x⁺,x,u,z,t)
        dyn_x⁺(z) = discrete_dynamics(model,z,x,u,h,t)

        r_idx = (t-1)*nx .+ (1:nx)

        ∇c[r_idx,idx.x[t]] = ForwardDiff.jacobian(dyn_x,x)
        ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        ∇c[r_idx,idx.h[t]] = ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t]))
        ∇c[r_idx,idx.x[t+1]] = ForwardDiff.jacobian(dyn_x⁺,x⁺)

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            r_idx = nx*(T-1) + t
            ∇c[r_idx,idx.h[t]] = -1.0
            ∇c[r_idx,idx.h[t+1]] = 1.0
        end
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
        h = Z[idx.h[t]]
        x⁺ = Z[idx.x[t+1]]

        dyn_x(z) = discrete_dynamics(model,x⁺,z,u,h,t)
        dyn_u(z) = discrete_dynamics(model,x⁺,x,z,h,t)
        dyn_h(z) = discrete_dynamics(model,x⁺,x,u,z,t)
        dyn_x⁺(z) = discrete_dynamics(model,z,x,u,h,t)

        r_idx = (t-1)*nx .+ (1:nx)

        s = nx*nx
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x,x))
        shift += s

        # ∇c[r_idx,idx.u[t]] = ForwardDiff.jacobian(dyn_u,u)
        s = nx*nu
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u,u))
        shift += s

        # ∇c[r_idx,idx.h[t]] = ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t]))
        s = nx*1
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t])))
        shift += s

        # ∇c[r_idx,idx.x[t+1]] .= ForwardDiff.jacobian(dyn_x,x)
        s = nx*nx
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x⁺,x⁺))
        shift += s

        if t < T-1
            h⁺ = Z[idx.h[t+1]]
            # r_idx = p_dyn + t
            # ∇c[r_idx,idx.h[t]] = -1.0
            s = 1
            ∇c[shift + s] = -1.0
            shift += s

            # ∇c[r_idx,idx.h[t+1]] = 1.0
            s = 1
            ∇c[shift + s] = 1.0
            shift += s
        end
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
        row_col!(row,col,r_idx,idx.h[t])
        row_col!(row,col,r_idx,idx.x[t+1])

        if t < T-1
            r_idx = r_shift + nx*(T-1) + t
            row_col!(row,col,r_idx,idx.h[t])
            row_col!(row,col,r_idx,idx.h[t+1])
        end
    end

    return collect(zip(row,col))
end
