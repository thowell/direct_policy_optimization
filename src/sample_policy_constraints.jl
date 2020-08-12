function sample_policy_constraints!(c,z,prob::SampleProblem)
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
    nu = length(idx_nom.u[1])
    nu_policy = length(u_policy)

    # controller for samples
    for t = 1:T-2
        x_nom = view(z,idx_nom.x[t+2])
        u_nom = view(z,idx_nom.u[t][u_policy])
        K = view(z,idx_K[t])

        for i = 1:N
            xi = view(z,idx_sample[i].x[t+2])
            ui = view(z,idx_sample[i].u[t][u_policy])
            ūi = view(z,idx_sample[i].u[t][(nu_policy+1):nu])
            # c[shift .+ (1:nu_policy)] = ui + K*(xi - x_nom) - u_nom
            c[shift .+ (1:nu_policy)] = policy(prob.models[i],K,xi,ūi,x_nom,u_nom) - ui

            shift += nu_policy
        end
    end

    nothing
end

function ∇sample_policy_constraints!(∇c,z,prob::SampleProblem)
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
    nu = length(idx_nom.u[1])
    nu_policy = length(u_policy)

    Im = Diagonal(ones(nu_policy))

    s = 0

    # policy for samples
    for t = 1:T-2
        x_nom = view(z,idx_nom.x[t+2])
        u_nom = view(z,idx_nom.u[t][u_policy])
        K = view(z,idx_K[t])

        for i = 1:N
            xi = view(z,idx_sample[i].x[t+2])
            ui = view(z,idx_sample[i].u[t][u_policy])
            ūi = view(z,idx_sample[i].u[t][(nu_policy+1):nu])

            # c[shift .+ (1:nu_policy)] = policy(prob.models[i],K,xi,ūi,x_nom,u_nom) - ui

            pK(y) = policy(prob.models[i],y,xi,ūi,x_nom,u_nom)
            pxi(y) = policy(prob.models[i],K,y,ūi,x_nom,u_nom)
            pūi(y) = policy(prob.models[i],K,xi,y,x_nom,u_nom)
            px_nom(y) = policy(prob.models[i],K,xi,ūi,y,u_nom)
            pu_nom(y) = policy(prob.models[i],K,xi,ūi,x_nom,y)

            r_idx = shift .+ (1:nu_policy)

            c_idx = idx_K[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pK,K))
            s += len

            c_idx = idx_sample[i].x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pxi,xi))
            s += len

            c_idx = idx_sample[i].u[t][(nu_policy+1):nu]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pūi,ūi))
            s += len

            c_idx = idx_nom.x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px_nom,x_nom))
            s += len

            c_idx = idx_nom.u[t][u_policy]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pu_nom,u_nom))
            s += len

            c_idx = idx_sample[i].u[t][u_policy]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(-Im)
            s += len

            shift += nu_policy
        end
    end

    nothing
end

function sparsity_jacobian_sample_policy(prob::SampleProblem;
        r_shift=0)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    idx_x_tmp = prob.idx_x_tmp
    idx_K = prob.idx_K
    idx_uw = prob.idx_uw
    u_policy = prob.u_policy
    models = prob.models

    T = prob.prob.T
    N = prob.N

    shift = 0

    nx = length(idx_nom.x[1])
    nu = length(idx_nom.u[1])
    nu_policy = length(u_policy)

    s = 0

    row = []
    col = []

    # controller for samples
    for t = 1:T-2
        for i = 1:N

            r_idx = r_shift + shift .+ (1:nu_policy)

            c_idx = idx_K[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t+2]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t][(nu_policy+1):nu]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.x[t+2]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.u[t][u_policy]
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t][u_policy]
            row_col!(row,col,r_idx,c_idx)

            shift += nu_policy
        end
    end

    return collect(zip(row,col))
end
