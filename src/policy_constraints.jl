function policy_constraints!(c,z,prob::SampleProblem)
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

    # nx = length(idx_nom.x[1])
    nu = length(idx_nom.u[1])
    nu_policy = length(u_policy)

    # controller for samples
    for t = 1:T-1
        x_nom = view(z,idx_nom.x[t])
        u_nom = view(z,idx_nom.u[t])
        K = view(z,idx_K[t])

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            c[shift .+ (1:nu_policy)] = (prob.policy_constraint ? policy(prob.models[i],K,xi,ui,x_nom,u_nom) : no_policy(prob.models[i],K,xi,ui,x_nom,u_nom)) - ui[u_policy]
            shift += nu_policy
        end
    end

    nothing
end

function ∇policy_constraints!(∇c,z,prob::SampleProblem)
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
    # nx = length(idx_nom.x[1])
    nu = length(idx_nom.u[1])
    nu_policy = length(u_policy)

    Im = Diagonal(ones(nu_policy))

    s = 0

    # policy for samples
    for t = 1:T-1
        x_nom = view(z,idx_nom.x[t])
        u_nom = view(z,idx_nom.u[t])
        K = view(z,idx_K[t])

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])

            pK(y) = policy(prob.models[i],y,xi,ui,x_nom,u_nom)
            pxi(y) = policy(prob.models[i],K,y,ui,x_nom,u_nom)
            pui(y) = policy(prob.models[i],K,xi,y,x_nom,u_nom) - y[u_policy]
            px_nom(y) = policy(prob.models[i],K,xi,ui,y,u_nom)
            pu_nom(y) = policy(prob.models[i],K,xi,ui,x_nom,y)

            r_idx = shift .+ (1:nu_policy)

            c_idx = idx_K[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pK,K))
            s += len

            c_idx = idx_sample[i].x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pxi,xi))
            s += len

            c_idx = idx_sample[i].u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pui,ui))
            s += len

            c_idx = idx_nom.x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px_nom,x_nom))
            s += len

            c_idx = idx_nom.u[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pu_nom,u_nom))
            s += len

            shift += nu_policy
        end
    end

    nothing
end

function sparsity_jacobian_policy(prob::SampleProblem;
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

    # nx = length(idx_nom.x[1])
    nu = length(idx_nom.u[1])
    nu_policy = length(u_policy)

    s = 0

    row = []
    col = []

    if prob.policy_constraint
        # controller for samples
        for t = 1:T-1
            for i = 1:N

                r_idx = r_shift + shift .+ (1:nu_policy)

                c_idx = idx_K[t]
                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_sample[i].x[t]
                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_sample[i].u[t]
                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_nom.x[t]
                row_col!(row,col,r_idx,c_idx)

                c_idx = idx_nom.u[t]
                row_col!(row,col,r_idx,c_idx)

                shift += nu_policy
            end
        end
    else
        r_idx = r_shift .+ (1:0)
        c_idx = (1:0)
        row_col!(row,col,r_idx,c_idx)
    end

    return collect(zip(row,col))
end
