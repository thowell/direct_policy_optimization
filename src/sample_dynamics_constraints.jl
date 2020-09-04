function sample_dynamics_constraints!(c,z,prob::SampleProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_policy = prob.u_policy
    models = prob.models
    β = prob.β
    w = prob.w
    m_stage = prob.prob.m_stage
    T = prob.prob.T
    N = prob.N
    disturbance_ctrl = prob.disturbance_ctrl

    shift = 0

    nx = length(idx_nom.x[1])
    nu = length(u_policy)

    # dynamics + resampling (x1 is taken care of w/ primal bounds)
    for t = 1:T-1
        x⁺_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        xs⁺ = (t in prob.resample_idx ? resample(x⁺_tmp,β=β,w=w) : x⁺_tmp) # resample

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            hi = view(z,idx_sample[i].h[t])

            xi⁺ = view(z,idx_sample[i].x[t+1])

            c[shift .+ (1:nx)] = discrete_dynamics(models[i],x⁺_tmp[i],xi,ui,hi,t)
            shift += nx
            c[shift .+ (1:nx)] = xs⁺[i] - xi⁺

            if disturbance_ctrl
                uwi = view(z,idx_uw[i][t])
                c[shift .+ (1:nx)] += uwi
            end

            shift += nx

            if t < T-1
                hi⁺ = view(z,idx_sample[i].h[t+1])

                c[shift + 1] = hi⁺[1] - hi[1]

                shift += 1
            end
        end
    end
    nothing
end

function ∇sample_dynamics_constraints!(∇c,z,prob::SampleProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_policy = prob.u_policy
    models = prob.models
    β = prob.β
    w = prob.w
    m_stage = prob.prob.m_stage
    T = prob.prob.T
    N = prob.N
    disturbance_ctrl = prob.disturbance_ctrl

    shift = 0
    nx = length(idx_nom.x[1])
    nu = length(u_policy)

    # dynamics + resampling (x1 is taken care of w/ primal bounds)
    s = 0
    for t = 1:T-1
        x⁺_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        x⁺_tmp_vec = vcat(x⁺_tmp...)
        idx_x_tmp_vec = vcat([idx_x_tmp[i].x[t] for i = 1:N]...)
        xs⁺ = (t in prob.resample_idx ? resample(x⁺_tmp,β=β,w=w) : x⁺_tmp )# resample

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            hi = z[idx_sample[i].h[t]]
            xi⁺ = view(z,idx_sample[i].x[t+1])

            dyn_x(a) = discrete_dynamics(models[i],x⁺_tmp[i],a,ui,hi,t)
            dyn_u(a) = discrete_dynamics(models[i],x⁺_tmp[i],xi,a,hi,t)
            dyn_h(a) = discrete_dynamics(models[i],x⁺_tmp[i],xi,ui,a,t)
            dyn_x_tmp(a) = discrete_dynamics(models[i],a,xi,ui,hi,t)
            function resample_x_tmp(a)
                if t in prob.resample_idx
                    return resample_vec(a,nx,N,i,β=β,w=w)
                else
                    return a[(i-1)*nx .+ (1:nx)]
                end
            end

            # c[shift .+ (1:nx)] = integration(models[i],x⁺_tmp[i],xi,ui,h)
            r_idx = shift .+ (1:nx)

            c_idx = idx_sample[i].x[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_x,xi)
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x,xi))
            s += len

            c_idx = idx_sample[i].u[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_u,ui)
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_u,ui))
            s += len

            c_idx = idx_sample[i].h[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_h,view(z,idx_nom.h[t]))
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_h,view(z,idx_sample[i].h[t])))
            s += len

            c_idx = idx_x_tmp[i].x[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_x_tmp,x⁺_tmp[i])
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x_tmp,x⁺_tmp[i]))
            s += len

            shift += nx

            # c[shift .+ (1:nx)] = xs⁺[i] - xi⁺
            r_idx = shift .+ (1:nx)

            c_idx = idx_sample[i].x[t+1]
            # ∇c[r_idx,c_idx] = Diagonal(-1.0*ones(nx))
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
            s += len

            c_idx = idx_x_tmp_vec
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(resample_x_tmp,x⁺_tmp_vec)
            len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(resample_x_tmp,x⁺_tmp_vec))
            ∇c[s .+ (1:len)] = vec(real.(FiniteDiff.finite_difference_jacobian(resample_x_tmp,x⁺_tmp_vec)))

            s += len

            if disturbance_ctrl
                # uwi = view(z,idx_sample[i].u[t][nu .+ (1:nx)])
                # c[shift .+ (1:nx)] += uwi
                c_idx = idx_uw[i][t]
                len = length(r_idx)*length(c_idx)
                ∇c[s .+ (1:len)] = vec(Diagonal(ones(nx)))
                s += len
            end

            shift += nx

            if t < T-1
                r_idx = shift .+ (1:1)

                c_idx = idx_sample[i].h[t+1]

                len = length(r_idx)*length(c_idx)
                ∇c[s + 1] = 1.0
                s += len

                c_idx = idx_sample[i].h[t]
                len = length(r_idx)*length(c_idx)
                ∇c[s + 1] = -1.0
                s += len

                shift += 1
            end
        end
    end
    nothing
end

function sparsity_jacobian_sample_dynamics(prob::SampleProblem;
        r_shift=0)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_policy = prob.u_policy
    models = prob.models
    β = prob.β
    w = prob.w
    m_stage = prob.prob.m_stage
    T = prob.prob.T
    N = prob.N
    disturbance_ctrl = prob.disturbance_ctrl

    shift = 0

    nx = length(idx_nom.x[1])
    nu = length(u_policy)

    s = 0

    row = []
    col = []

    # dynamics + resampling (x1 is taken care of w/ primal bounds)
    for t = 1:T-1
        # h = z[idx_nom.h[t]]
        # x⁺_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        # x⁺_tmp_vec = vcat(x⁺_tmp...)
        idx_x_tmp_vec = vcat([idx_x_tmp[i].x[t] for i = 1:N]...)
        # xs⁺ = resample(x⁺_tmp,β=β,w=w) # resample

        for i = 1:N
            # xi = view(z,idx_sample[i].x[t])
            # ui = view(z,idx_sample[i].u[t])
            # xi⁺ = view(z,idx_sample[i].x[t+1])
            #
            # dyn_x(a) = integration(models[i],x⁺_tmp[i],a,ui,h)
            # dyn_u(a) = integration(models[i],x⁺_tmp[i],xi,a,h)
            # dyn_h(a) = integration(models[i],x⁺_tmp[i],xi,ui,a)
            # dyn_x_tmp(a) = integration(models[i],a,xi,ui,h)
            # resample_x_tmp(a) = resample_vec(a,nx,N,i,β=β,w=w) # resample

            # c[shift .+ (1:nx)] = integration(models[i],x⁺_tmp[i],xi,ui,h)
            r_idx = r_shift + shift .+ (1:nx)

            c_idx = idx_sample[i].x[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_x,xi)
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x,xi))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_u,ui)
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_u,ui))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].h[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_h,view(z,idx_nom.h[t]))
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_h,view(z,idx_nom.h[t])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_x_tmp[i].x[t]
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(dyn_x_tmp,x⁺_tmp[i])
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x_tmp,x⁺_tmp[i]))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            shift += nx

            # c[shift .+ (1:nx)] = xs⁺[i] - xi⁺
            r_idx = r_shift + shift .+ (1:nx)

            c_idx = idx_sample[i].x[t+1]
            # ∇c[r_idx,c_idx] = Diagonal(-1.0*ones(nx))
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_x_tmp_vec
            # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(resample_x_tmp,x⁺_tmp_vec)
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(resample_x_tmp,x⁺_tmp_vec))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            if disturbance_ctrl
                # uwi = view(z,idx_sample[i].u[t][nu .+ (1:nx)])
                # c[shift .+ (1:nx)] += uwi
                c_idx = idx_uw[i][t]
                # len = length(r_idx)*length(c_idx)
                # ∇c[s .+ (1:len)] = vec(Diagonal(ones(nx)))
                # s += len
                row_col!(row,col,r_idx,c_idx)
            end

            shift += nx

            if t < T-1
                r_idx = r_shift + shift .+ (1:1)

                c_idx = idx_sample[i].h[t+1]

                # len = length(r_idx)*length(c_idx)
                # ∇c[s + 1] = 1.0
                # s += len

                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_sample[i].h[t]
                # len = length(r_idx)*length(c_idx)
                # ∇c[s + 1] = -1.0
                # s += len

                row_col!(row,col,r_idx,c_idx)

                shift += 1
            end
        end
    end

    return collect(zip(row,col))
end
