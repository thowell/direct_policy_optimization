function contact_dynamics_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model
    m_contact = model.m_contact

    for t = 1:T-2
        x2 = Z[idx.x[t+1]]
        x3 = Z[idx.x[t+2]]
        u = Z[idx.u[t]]
        h = Z[idx.h[t]]

        c[(t-1)*m_contact .+ (1:model.nb)] = maximum_energy_dissipation(model,x2,x3,u,h)
        c[(t-1)*m_contact+model.nb .+ (1:model.nc)] = ϕ_func(model,x3)
        c[(t-1)*m_contact+model.nb + model.nc .+ (1:model.nc)] = friction_cone(model,u)
        c[(t-1)*m_contact+model.nb + 2*model.nc + 1] = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,x3)
        c[(t-1)*m_contact+model.nb + 2*model.nc + 2] = u[model.idx_s] - u[model.idx_ψ]'*friction_cone(model,u)
        c[(t-1)*m_contact+model.nb + 2*model.nc + 3] = u[model.idx_s] - u[model.idx_b]'*u[model.idx_η]
    end

    c[(T-2)*m_contact .+ (1:model.nc)] = ϕ_func(model,Z[idx.x[1]])
    c[(T-2)*m_contact+model.nc .+ (1:model.nc)] = ϕ_func(model,Z[idx.x[2]])

    return nothing
end

function contact_dynamics_constraints_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model
    m_contact = model.m_contact

    shift = 0

    for t = 1:T-2
        x2 = Z[idx.x[t+1]]
        x3 = Z[idx.x[t+2]]
        u = Z[idx.u[t]]
        h = Z[idx.h[t]]

        medx2(z) = maximum_energy_dissipation(model,z,x3,u,h,t)
        medx3(z) = maximum_energy_dissipation(model,x2,z,u,h,t)
        medu(z) = maximum_energy_dissipation(model,x2,x3,z,h,t)
        medh(z) = maximum_energy_dissipation(model,x2,x3,u,z,t)
        r_idx = (t-1)*m_contact .+ (1:model.nb)

        c_idx = idx.x[t+1]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medx2,x2))
        shift += len

        c_idx = idx.x[t+2]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medx3,x3))
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medu,u))
        shift += len

        c_idx = idx.h[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medh,view(Z,idx.h[t])))
        shift += len

        #
        ϕx3(z) = ϕ_func(model,z)
        r_idx = (t-1)*m_contact+model.nb .+ (1:model.nc)

        c_idx = idx.x[t+2]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx3,x3))
        shift += len

        #
        fcu(z) = friction_cone(model,z)
        r_idx = (t-1)*m_contact+model.nb + model.nc .+ (1:model.nc)

        c_idx = idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(fcu,u))
        shift += len


        comp1_x3(z) = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,z)
        comp1_u(z) = z[model.idx_s] - z[model.idx_λ]'*ϕ_func(model,x3)
        r_idx = (t-1)*m_contact+model.nb + 2*model.nc + 1

        c_idx = idx.x[t+2]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_x3,x3)
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_u,u)
        shift += len


        comp2_u(z) = z[model.idx_s] - z[model.idx_ψ]'*friction_cone(model,z)
        r_idx = (t-1)*m_contact + model.nb + 2*model.nc + 2

        c_idx = idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp2_u,u)
        shift += len

        comp3_u(z) = z[model.idx_s] - z[model.idx_b]'*z[model.idx_η]
        r_idx = (t-1)*m_contact+model.nb + 2*model.nc + 3

        c_idx = idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp3_u,u)
        shift += len
    end

    ϕx(z) = ϕ_func(model,x)
    # c[(T-2)*m_contact .+ (1:model.nc)] = ϕ_func(model,Z[idx.x[1]])
    r_idx = (T-2)*m_contact .+ (1:model.nc)
    c_idx = idx.x[1]

    len = length(r_idx)*length(c_idx)
    ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx,Z[idx.x[1]]))
    shift += len

    # c[(T-2)*m_contact+model.nc .+ (1:model.nc)] = ϕ_func(model,Z[idx.x[2]])
    r_idx = (T-2)*m_contact+model.nc .+ (1:model.nc)
    c_idx = idx.x[2]

    len = length(r_idx)*length(c_idx)
    ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx,Z[idx.x[2]]))
    shift += len

    return nothing
end

function sparsity_contact_dynamics_jacobian(prob::TrajectoryOptimizationProblem;
        r_shift=0)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model
    m_contact = model.m_contact
    row = []
    col = []

    for t = 1:T-2
        # x2 = Z[idx.x[t+1]]
        # x3 = Z[idx.x[t+2]]
        # u = Z[idx.u[t]]
        # h = Z[idx.h[t]]

        # medx2(z) = maximum_energy_dissipation(model,z,x2,u,h,t)
        # medx3(z) = maximum_energy_dissipation(model,x2,z,u,h,t)
        # medu(z) = maximum_energy_dissipation(model,x2,x2,z,h,t)
        # medh(z) = maximum_energy_dissipation(model,x2,x2,u,z,t)
        r_idx = r_shift + (t-1)*m_contact .+ (1:model.nb)

        c_idx = idx.x[t+1]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medx2,x2))
        # shift += len
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.x[t+2]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medx3,x3))
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        c_idx = idx.u[t]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medu,u))
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        c_idx = idx.h[t]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medh,view(Z,idx.h[t])))
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        #
        # ϕx3(z) = ϕ_func(model,z)
        r_idx = r_shift + (t-1)*m_contact+model.nb .+ (1:model.nc)

        c_idx = idx.x[t+2]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx3,x3))
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        # fcu(z) = friction_cone(model,z)
        r_idx = r_shift + (t-1)*m_contact+model.nb + model.nc .+ (1:model.nc)

        c_idx = idx.u[t]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(fcu,u))
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        # comp1_x3(z) = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,z)
        # comp1_u(z) = z[model.idx_s] - z[model.idx_λ]'*ϕ_func(model,x3)
        r_idx = r_shift + (t-1)*m_contact+model.nb + 2*model.nc + 1

        c_idx = idx.x[t+2]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_x3,x3)
        # shift += len
        row_col!(row,col,r_idx,c_idx)

        c_idx = idx.u[t]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_u,u)
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        # comp2_u(z) = u[model.idx_s] - u[model.idx_ψ]'*friction_cone(model,u)
        r_idx = r_shift + (t-1)*m_contact+model.nb + 2*model.nc + 2

        c_idx = idx.u[t]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp2_u,u)
        # shift += len
        row_col!(row,col,r_idx,c_idx)


        # comp3_u(z) = u[model.idx_s] - u[model.idx_b]'*u[model.idx_η]
        r_idx = r_shift + (t-1)*m_contact+model.nb + 2*model.nc + 3

        c_idx = idx.u[t]
        # len = length(r_idx)*length(c_idx)
        # ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp3_u,u)
        # shift += len
        row_col!(row,col,r_idx,c_idx)
    end

    # ϕx(z) = ϕ_func(model,x)
    # c[(T-2)*m_contact .+ (1:model.nc)] = ϕ_func(model,Z[idx.x[1]])
    r_idx = (T-2)*m_contact .+ (1:model.nc)

    c_idx = idx.x[1]
    row_col!(row,col,r_idx,c_idx)

    # len = length(r_idx)*length(c_idx)
    # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx,Z[idx.x[1]]))
    # shift += len

    # c[(T-2)*m_contact+model.nc .+ (1:model.nc)] = ϕ_func(model,Z[idx.x[2]])
    r_idx = (T-2)*m_contact+model.nc .+ (1:model.nc)
    c_idx = idx.x[2]
    row_col!(row,col,r_idx,c_idx)

    # len = length(r_idx)*length(c_idx)
    # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx,Z[idx.x[2]]))
    # shift += len

    return collect(zip(row,col))
end
