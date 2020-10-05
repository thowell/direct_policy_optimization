function sample_stage_constraints!(c,z,prob::DPOProblem)
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

    # stage constraints samples
    for t = 1:T-1
        if m_stage[t] > 0
            for i = 1:N
                xi = view(z,idx_sample[i].x[t])
                ui = view(z,idx_sample[i].u[t])
                c_stage!(view(c,shift .+ (1:m_stage[t])),xi,ui,t,models[i])
                shift += m_stage[t]
            end
        end
    end
    nothing
end

function ∇sample_stage_constraints!(∇c,z,prob::DPOProblem)
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

    # stage constraints samples
    for t = 1:T-1
        if m_stage[t] > 0
            c_stage_tmp = zeros(m_stage[t])
            for i = 1:N
                xi = view(z,idx_sample[i].x[t])
                ui = view(z,idx_sample[i].u[t])

                # con(view(c,shift .+ (1:m_stage)),xi,ui)

                cx(c,a) = c_stage!(c,a,ui,t,models[i])
                cu(c,a) = c_stage!(c,xi,a,t,models[i])

                r_idx = shift .+ (1:m_stage[t])

                c_idx = idx_sample[i].x[t]
                # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(con_x,c_stage_tmp,xi)
                len = length(r_idx)*length(c_idx)
                ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(cx,c_stage_tmp,xi))
                s += len

                c_idx = idx_sample[i].u[t]
                # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(con_u,c_stage_tmp,ui)
                len = length(r_idx)*length(c_idx)
                ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(cu,c_stage_tmp,ui))
                s += len

                shift += m_stage[t]
            end
        end
    end
    nothing
end

function sparsity_jacobian_sample_stage(prob::DPOProblem;
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

    # stage constraints samples

    for t = 1:T-1
        c_stage_tmp = zeros(m_stage[t])
        if m_stage[t] > 0
            for i = 1:N
                # xi = view(z,idx_sample[i].x[t])
                # ui = view(z,idx_sample[i].u[t])
                #
                # # con(view(c,shift .+ (1:m_stage)),xi,ui)
                #
                # con_x(c,a) = con(c,a,ui)
                # con_u(c,a) = con(c,xi,a)

                r_idx = r_shift + shift .+ (1:m_stage[t])

                c_idx = idx_sample[i].x[t]
                # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(con_x,c_stage_tmp,xi)
                len = length(r_idx)*length(c_idx)
                # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(con_x,c_stage_tmp,xi))
                s += len
                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_sample[i].u[t]
                # ∇c[r_idx,c_idx] = ForwardDiff.jacobian(con_u,c_stage_tmp,ui)
                len = length(r_idx)*length(c_idx)
                # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(con_u,c_stage_tmp,ui))
                s += len
                row_col!(row,col,r_idx,c_idx)

                shift += m_stage[t]
            end
        end
    end
    return collect(zip(row,col))
end
