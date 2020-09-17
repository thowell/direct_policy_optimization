function dynamics_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    for t = 1:T-2
        x1 = Z[idx.x[t]]
        x2 = Z[idx.x[t+1]]
        x3 = Z[idx.x[t+2]]
        u = Z[idx.u[t]]
        h = Z[idx.h[t]]

        c[(t-1)*nx .+ (1:nx)] = discrete_dynamics(model,x1,x2,x3,u,h,t)

        if t < T-2
            h⁺ = Z[idx.h[t+1]]
            c[nx*(T-2) + t] = h⁺ - h
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

    shift = 0

    for t = 1:T-2
        x1 = Z[idx.x[t]]
        x2 = Z[idx.x[t+1]]
        x3 = Z[idx.x[t+2]]
        u = Z[idx.u[t]]
        h = Z[idx.h[t]]

        dyn_x1(z) = discrete_dynamics(model,z,x2,x3,u,h,t)
        dyn_x2(z) = discrete_dynamics(model,x1,z,x3,u,h,t)
        dyn_x3(z) = discrete_dynamics(model,x1,x2,z,u,h,t)
        dyn_u(z) = discrete_dynamics(model,x1,x2,x3,z,h,t)
        dyn_h(z) = discrete_dynamics(model,x1,x2,x3,u,z,t)

        r_idx = (t-1)*nx .+ (1:nx)

        c_idx = idx.x[t]
        s = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x1,x1))
        shift += s

        c_idx = idx.x[t+1]
        s = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x2,x2))
        shift += s

        c_idx = idx.x[t+2]
        s = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x3,x3))
        shift += s

        c_idx = idx.u[t]
        s = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u,u))
        shift += s

        c_idx = idx.h[t]
        s = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_h,[h]))#vec(ForwardDiff.jacobian(dyn_h,view(Z,idx.h[t])))
        shift += s

        if t < T-2
            r_idx = nx*(T-2) + t

            c_idx = idx.h[t]
            s = 1
            ∇c[shift + s] = -1.0
            shift += s

            c_idx = idx.h[t+1]
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

    for t = 1:T-2


        r_idx = r_shift + (t-1)*nx .+ (1:nx)

        c_idx = idx.x[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.x[t+1]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.x[t+2]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.u[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.h[t]
        row_col!(row,col,r_idx,c_idx)


        if t < T-2
            r_idx = r_shift + nx*(T-2) + t

            c_idx = idx.h[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx.h[t+1]
            row_col!(row,col,r_idx,c_idx)
        end
    end

    return collect(zip(row,col))
end
