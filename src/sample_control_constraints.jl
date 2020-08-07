function sample_control_constraints!(c,z,prob::SampleProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_ctrl = prob.u_ctrl
    models = prob.models
    β = prob.β
    w = prob.w
    m_stage = prob.prob.m_stage
    T = prob.prob.T
    N = prob.N
    disturbance_ctrl = prob.disturbance_ctrl

    shift = 0

    nx = length(idx_nom.x[1])
    nu_ctrl = length(u_ctrl)

    # controller for samples
    for t = 1:T-1
        x_nom = view(z,idx_nom.x[t])
        u_nom = view(z,idx_nom.u[t][u_ctrl])
        K = reshape(view(z,idx_K[t]),nu_ctrl,nx)

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t][u_ctrl])
            c[shift .+ (1:nu_ctrl)] = ui + K*(xi - x_nom) - u_nom
            shift += nu_ctrl
        end
    end

    nothing
end

function ∇sample_control_constraints!(∇c,z,prob::SampleProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_ctrl = prob.u_ctrl
    models = prob.models
    β = prob.β
    w = prob.w
    m_stage = prob.prob.m_stage
    T = prob.prob.T
    N = prob.N
    disturbance_ctrl = prob.disturbance_ctrl

    shift = 0
    nx = length(idx_nom.x[1])
    nu_ctrl = length(u_ctrl)

    s = 0

    # controller for samples
    Im = Diagonal(ones(nu_ctrl))
    for t = 1:T-1
        x_nom = view(z,idx_nom.x[t])
        u_nom = view(z,idx_nom.u[t][u_ctrl])
        K = reshape(view(z,idx_K[t]),nu_ctrl,nx)

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t][u_ctrl])
            # c[shift .+ (1:nu)] = ui + K*(xi - x_nom) - u_nom

            r_idx = shift .+ (1:nu_ctrl)

            c_idx = idx_nom.x[t]
            # ∇c[r_idx,c_idx] = -1.0*K
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(-1.0*K)
            s += len

            c_idx = idx_nom.u[t][u_ctrl]
            # ∇c[r_idx,c_idx] = Diagonal(-1.0*ones(nu))
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nu)))
            s += len

            c_idx = idx_sample[i].x[t]
            # ∇c[r_idx,c_idx] = K
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(K)
            s += len

            c_idx = idx_sample[i].u[t][u_ctrl]
            # ∇c[r_idx,c_idx] = Diagonal(ones(nu))
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(ones(nu)))
            s += len

            c_idx = idx_K[t]
            # ∇c[r_idx,c_idx] = kron((xi - x_nom)',Im)
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(kron((xi - x_nom)',Im))
            s += len

            shift += nu_ctrl
        end
    end

    nothing
end

function sparsity_jacobian_sample_control(prob::SampleProblem;
        r_shift=0)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_ctrl = prob.u_ctrl
    models = prob.models
    β = prob.β
    w = prob.w
    m_stage = prob.prob.m_stage
    T = prob.prob.T
    N = prob.N
    disturbance_ctrl = prob.disturbance_ctrl

    shift = 0

    nx = length(idx_nom.x[1])
    nu_ctrl = length(u_ctrl)

    s = 0

    row = []
    col = []

    # controller for samples
    for t = 1:T-1
        # x_nom = view(z,idx_nom.x[t])
        # u_nom = view(z,idx_nom.u[t])
        # K = reshape(view(z,idx_K[t]),nu,nx)

        for i = 1:N
            # xi = view(z,idx_sample[i].x[t])
            # ui = view(z,idx_sample[i].u[t])
            # c[shift .+ (1:nu)] = ui + K*(xi - x_nom) - u_nom

            r_idx = r_shift + shift .+ (1:nu_ctrl)

            c_idx = idx_nom.x[t]
            # ∇c[r_idx,c_idx] = -1.0*K
            len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(-1.0*K)
            s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.u[t][u_ctrl]
            # ∇c[r_idx,c_idx] = Diagonal(-1.0*ones(nu))
            len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nu)))
            s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t]
            # ∇c[r_idx,c_idx] = K
            len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(K)
            s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t][u_ctrl]
            # ∇c[r_idx,c_idx] = Diagonal(ones(nu))
            len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(Diagonal(ones(nu)))
            s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_K[t]
            # ∇c[r_idx,c_idx] = kron((xi - x_nom)',Im)
            len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(kron((xi - x_nom)',Im))
            s += len
            row_col!(row,col,r_idx,c_idx)

            shift += nu_ctrl
        end
    end

    return collect(zip(row,col))
end
