function contact_dynamics_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model
    m_contact = model.m_contact

    for t = 1:T-2
        x3 = Z[idx.x[t+2]]
        u = Z[idx.u[t]]

        c[(t-1)*m_contact .+ (1:model.nc)] = ϕ_func(model,x3)
        c[(t-1)*m_contact + model.nc + 1] = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,x3)
    end

    return nothing
end

function contact_dynamics_constraints_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    shift = 0

    for t = 1:T-2
        x3 = Z[idx.x[t+2]]
        u = Z[idx.u[t]]

        # c[(t-1)*m_contact .+ (1:model.nc)] = ϕ_func(model,x3)
        ϕ(z) = ϕ_func(model,z)
        r_idx = (t-1)*m_contact .+ (1:model.nc)

        c_idx = idx.x[t+2]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕ,x3))
        shift += len

        # c[(t-1)*m_contact + model.nc + 1] = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,x3)
        comp1_x3(z) = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,z)
        comp1_u(z) = z[model.idx_s] - z[model.idx_λ]'*ϕ_func(model,x3)
        r_idx = (t-1)*m_contact + model.nc + 1

        c_idx = idx.x[t+2]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_x3,x3)
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_u,u)
        shift += len
    end

    return nothing
end

function sparsity_contact_dynamics_jacobian(prob::TrajectoryOptimizationProblem;
        r_shift=0)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model

    row = []
    col = []

    for t = 1:T-2

        r_idx = r_shift + (t-1)*m_contact .+ (1:model.nc)

        c_idx = idx.x[t+2]
        row_col!(row,col,r_idx,c_idx)

        r_idx = r_shift + (t-1)*m_contact + model.nc + 1

        c_idx = idx.x[t+2]
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.u[t]
        row_col!(row,col,r_idx,c_idx)

    end

    return collect(zip(row,col))
end
