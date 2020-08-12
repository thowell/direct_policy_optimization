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

    # dynamics + resampling (x1 and x2 are taken care of w/ primal bounds)
    for t = 1:T-2
        x3_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        x3s = resample(x3_tmp,β=β,w=w) # resample

        for i = 1:N
            x1i = view(z,idx_sample[i].x[t])
            x2i = view(z,idx_sample[i].x[t+1])
            x3i = view(z,idx_sample[i].x[t+2])

            ui = view(z,idx_sample[i].u[t])
            hi = view(z,idx_sample[i].h[t])

            c[shift .+ (1:nx)] = discrete_dynamics(models[i],x1i,x2i,x3_tmp[i],ui,hi,t)
            shift += nx
            c[shift .+ (1:nx)] = x3s[i] - x3i

            if disturbance_ctrl
                uwi = view(z,idx_uw[i][t])
                c[shift .+ (1:nx)] += uwi
            end

            shift += nx

            if t < T-2
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
    for t = 1:T-2
        x3_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        x3_tmp_vec = vcat(x3_tmp...)
        idx_x_tmp_vec = vcat([idx_x_tmp[i].x[t] for i = 1:N]...)
        x3s = resample(x3_tmp,β=β,w=w) # resample

        for i = 1:N
            x1i = view(z,idx_sample[i].x[t])
            x2i = view(z,idx_sample[i].x[t+1])
            x3i = view(z,idx_sample[i].x[t+2])
            ui = view(z,idx_sample[i].u[t])
            hi = z[idx_sample[i].h[t]]

            dyn_x1(a) = discrete_dynamics(models[i],a,x2i,x3_tmp[i],ui,hi,t)
            dyn_x2(a) = discrete_dynamics(models[i],x1i,a,x3_tmp[i],ui,hi,t)
            dyn_x3(a) = discrete_dynamics(models[i],x1i,x2i,a,ui,hi,t)
            dyn_u(a) = discrete_dynamics(models[i],x1i,x2i,x3_tmp[i],a,hi,t)
            dyn_h(a) = discrete_dynamics(models[i],x1i,x2i,x3_tmp[i],ui,a,t)
            resample_x3_tmp(a) = resample_vec(a,nx,N,i,β=β,w=w) # resample

            # c[shift .+ (1:nx)] = integration(models[i],x⁺_tmp[i],xi,ui,h)
            r_idx = shift .+ (1:nx)

            c_idx = idx_sample[i].x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x1,x1i))
            s += len

            c_idx = idx_sample[i].x[t+1]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x2,x2i))
            s += len

            c_idx = idx_x_tmp[i].x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_x3,x3_tmp[i]))
            s += len

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_u,ui))
            s += len

            c_idx = idx_sample[i].h[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(dyn_h,view(z,idx_sample[i].h[t])))
            s += len

            shift += nx

            # c[shift .+ (1:nx)] = xs⁺[i] - xi⁺
            r_idx = shift .+ (1:nx)

            c_idx = idx_sample[i].x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
            s += len

            c_idx = idx_x_tmp_vec
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(resample_x3_tmp,x3_tmp_vec))
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

            if t < T-2
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
    for t = 1:T-2
        idx_x_tmp_vec = vcat([idx_x_tmp[i].x[t] for i = 1:N]...)

        for i = 1:N
            r_idx = r_shift + shift .+ (1:nx)

            c_idx = idx_sample[i].x[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t+1]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_x_tmp[i].x[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].h[t]
            row_col!(row,col,r_idx,c_idx)

            shift += nx

            # c[shift .+ (1:nx)] = xs⁺[i] - xi⁺
            r_idx = r_shift + shift .+ (1:nx)

            c_idx = idx_sample[i].x[t+2]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_x_tmp_vec
            row_col!(row,col,r_idx,c_idx)

            if disturbance_ctrl
                c_idx = idx_uw[i][t]
                row_col!(row,col,r_idx,c_idx)
            end

            shift += nx

            if t < T-2
                r_idx = r_shift + shift .+ (1:1)

                c_idx = idx_sample[i].h[t+1]
                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_sample[i].h[t]
                row_col!(row,col,r_idx,c_idx)

                shift += 1
            end
        end
    end

    return collect(zip(row,col))
end
