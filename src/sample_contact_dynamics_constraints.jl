function sample_contact_dynamics_constraints!(c,Z,prob::SampleProblem)
    T = prob.prob.T
    model = prob.prob.model
    M = prob.prob.M_contact_dynamics
    idx_sample = prob.idx_sample

    for i = 1:prob.N
        # signed distance function
        for t = 1:T
            x = Z[idx_sample[i].x[t]]
            c[(i-1)*M + (t-1)*model.nc .+ (1:model.nc)] = ϕ_func(model,x)
        end

        m_sdf = model.nc*T

        # maximum energy dissipation
        for t = 1:T-2
            x2 = Z[idx_sample[i].x[t+1]]
            x3 = Z[idx_sample[i].x[t+2]]
            u = Z[idx_sample[i].u[t]]
            h = Z[idx_sample[i].h[t]]

            c[(i-1)*M + m_sdf + (t-1)*model.nb .+ (1:model.nb)] = maximum_energy_dissipation(model,x2,x3,u,h)
        end

        m_med = model.nb*(T-2)

        # friction cone
        for t = 1:T-2
            u = Z[idx_sample[i].u[t]]

            c[(i-1)*M + m_sdf+m_med + (t-1)*model.nc .+ (1:model.nc)] = friction_cone(model,u)
        end

        m_fc = model.nc*(T-2)

        # complementarity
        for t = 1:T-2
            x3 = Z[idx_sample[i].x[t+2]]
            u = Z[idx_sample[i].u[t]]

            c[(i-1)*M + m_sdf+m_med+m_fc + (t-1)*3 + 1] = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,x3)
            c[(i-1)*M + m_sdf+m_med+m_fc + (t-1)*3 + 2] = u[model.idx_s] - u[model.idx_ψ]'*friction_cone(model,u)
            c[(i-1)*M + m_sdf+m_med+m_fc + (t-1)*3 + 3] = u[model.idx_s] - u[model.idx_b]'*u[model.idx_η]
        end
    end
    return nothing
end

function sample_contact_dynamics_constraints_jacobian!(∇c,Z,prob::SampleProblem)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model
    idx_sample = prob.idx_sample

    shift = 0

    for i = 1:N
        for t = 1:T
            x = Z[idx_sample[i].x[t]]

            ϕx(z) = ϕ_func(model,z)
            r_idx = (t-1)*model.nc .+ (1:model.nc)

            c_idx = idx_sample[i].x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(ϕx,x))
            shift += len
        end

        m_sdf = model.nc*T

        for t = 1:T-2
            x2 = Z[idx_sample[i].x[t+1]]
            x3 = Z[idx_sample[i].x[t+2]]
            u = Z[idx_sample[i].u[t]]
            h = Z[idx_sample[i].h[t]]

            medx2(z) = maximum_energy_dissipation(model,z,x3,u,h)
            medx3(z) = maximum_energy_dissipation(model,x2,z,u,h)
            medu(z) = maximum_energy_dissipation(model,x2,x3,z,h)
            medh(z) = maximum_energy_dissipation(model,x2,x3,u,z)
            r_idx = m_sdf + (t-1)*model.nb .+ (1:model.nb)

            c_idx = idx_sample[i].x[t+1]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medx2,x2))
            shift += len

            c_idx = idx_sample[i].x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medx3,x3))
            shift += len

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medu,u))
            shift += len

            c_idx = idx_sample[i].h[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(medh,view(Z,idx_sample[i].h[t])))
            shift += len
        end

        m_med = model.nb*(T-2)

        for t = 1:T-2
            u = Z[idx_sample[i].u[t]]
            #
            fcu(z) = friction_cone(model,z)
            r_idx = m_sdf + m_med + (t-1)*model.nc .+ (1:model.nc)

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(fcu,u))
            shift += len
        end

        m_fc = model.nc*(T-2)

        for t = 1:T-2
            x3 = Z[idx_sample[i].x[t+2]]
            u = Z[idx_sample[i].u[t]]

            comp1_x3(z) = u[model.idx_s] - u[model.idx_λ]'*ϕ_func(model,z)
            comp1_u(z) = z[model.idx_s] - z[model.idx_λ]'*ϕ_func(model,x3)
            r_idx = m_sdf + m_med + m_fc + (t-1)*3 + 1

            c_idx = idx_sample[i].x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_x3,x3)
            shift += len

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp1_u,u)
            shift += len

            comp2_u(z) = z[model.idx_s] - z[model.idx_ψ]'*friction_cone(model,z)
            r_idx = m_sdf + m_med + m_fc + (t-1)*3 + 2

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp2_u,u)
            shift += len

            comp3_u(z) = z[model.idx_s] - z[model.idx_b]'*z[model.idx_η]
            r_idx = m_sdf + m_med + m_fc + (t-1)*3 + 3

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = ForwardDiff.gradient(comp3_u,u)
            shift += len
        end
    end
    return nothing
end

function sparsity_sample_contact_dynamics_jacobian(prob::SampleProblem;
        r_shift=0)
    idx = prob.idx
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    model = prob.model
    M = prob.prob.M_contact_dynamics
    idx_sample = prob.idx_sample

    row = []
    col = []

    for i = 1:N
        for t = 1:T
            r_idx = r_shift + (i-1)*M + (t-1)*model.nc .+ (1:model.nc)
            c_idx = idx_sample[i].x[t]
            row_col!(row,col,r_idx,c_idx)
        end

        m_sdf = model.nc*T

        for t = 1:T-2
            r_idx = r_shift + (i-1)*M + m_sdf + (t-1)*model.nb .+ (1:model.nb)

            c_idx = idx_sample[i].x[t+1]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t+2]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].h[t]
            row_col!(row,col,r_idx,c_idx)
        end

        m_med = model.nb*(T-2)

        for t = 1:T-2
            r_idx = r_shift + (i-1)*M + m_sdf + m_med + (t-1)*model.nc .+ (1:model.nc)

            c_idx = idx_sample[i].u[t]
            row_col!(row,col,r_idx,c_idx)
        end

        m_fc = model.nc*(T-2)

        for t = 1:T-2
            r_idx = r_shift + (i-1)*M + m_sdf + m_med + m_fc + (t-1)*3 + 1

            c_idx = idx_sample[i].x[t+2]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t]
            row_col!(row,col,r_idx,c_idx)

            r_idx = r_shift + (i-1)*M + m_sdf + m_med + m_fc + (t-1)*3 + 2

            c_idx = idx_sample[i].u[t]
            row_col!(row,col,r_idx,c_idx)

            r_idx = r_shift + (i-1)*M + m_sdf + m_med + m_fc + (t-1)*3 + 3

            c_idx = idx_sample[i].u[t]
            row_col!(row,col,r_idx,c_idx)
        end
    end

    return collect(zip(row,col))
end
