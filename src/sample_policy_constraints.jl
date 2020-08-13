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
        x1_nom = view(z,idx_nom.x[t])
        x2_nom = view(z,idx_nom.x[t+1])
        x3_nom = view(z,idx_nom.x[t+2])
        u_nom = view(z,idx_nom.u[t][u_policy])
        h_nom = view(z,idx_nom.h[t])

        K = view(z,idx_K[t])

        for i = 1:N
            x1i = view(z,idx_sample[i].x[t])
            x2i = view(z,idx_sample[i].x[t+1])
            x3i = view(z,idx_sample[i].x[t+2])
            ui = view(z,idx_sample[i].u[t][u_policy])
            ūi = view(z,idx_sample[i].u[t][(nu_policy+1):nu])
            hi = view(z,idx_sample[i].h[t])
            # c[shift .+ (1:nu_policy)] = ui + K*(xi - x_nom) - u_nom
            c[shift .+ (1:nu_policy)] = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom) - ui

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
        x1_nom = view(z,idx_nom.x[t])
        x2_nom = view(z,idx_nom.x[t+1])
        x3_nom = view(z,idx_nom.x[t+2])
        u_nom = view(z,idx_nom.u[t][u_policy])
        h_nom = view(z,idx_nom.h[t])

        K = view(z,idx_K[t])

        for i = 1:N
            x1i = view(z,idx_sample[i].x[t])
            x2i = view(z,idx_sample[i].x[t+1])
            x3i = view(z,idx_sample[i].x[t+2])
            ui = view(z,idx_sample[i].u[t][u_policy])
            ūi = view(z,idx_sample[i].u[t][(nu_policy+1):nu])
            hi = z[idx_sample[i].h[t]]

            pK(y) = policy(prob.models[i],y,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            px1i(y) = policy(prob.models[i],K,y,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            px2i(y) = policy(prob.models[i],K,x1i,y,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            px3i(y) = policy(prob.models[i],K,x1i,x2i,y,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            pūi(y) = policy(prob.models[i],K,x1i,x2i,x3i,y,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            phi(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,y,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            px1_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,y,x2_nom,x3_nom,u_nom,h_nom)
            px2_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,y,x3_nom,u_nom,h_nom)
            px3_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,y,u_nom,h_nom)
            pu_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,y,h_nom)
            ph_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,y)

            r_idx = shift .+ (1:nu_policy)

            c_idx = idx_K[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pK,K))
            s += len

            c_idx = idx_sample[i].x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1i,x1i))
            s += len

            c_idx = idx_sample[i].x[t+1]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2i,x2i))
            s += len

            c_idx = idx_sample[i].x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3i,x3i))
            s += len

            c_idx = idx_sample[i].u[t][(nu_policy+1):nu]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pūi,ūi))
            s += len

            c_idx = idx_sample[i].h[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(phi,view(z,idx_sample[i].h[t])))
            s += len

            c_idx = idx_nom.x[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1_nom,x1_nom))
            s += len

            c_idx = idx_nom.x[t+1]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2_nom,x2_nom))
            s += len

            c_idx = idx_nom.x[t+2]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3_nom,x3_nom))
            s += len

            c_idx = idx_nom.u[t][u_policy]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pu_nom,u_nom))
            s += len

            c_idx = idx_nom.h[t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(phi,view(z,idx_nom.h[t])))
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

    for t = 1:T-2
        # x1_nom = view(z,idx_nom.x[t])
        # x2_nom = view(z,idx_nom.x[t+1])
        # x3_nom = view(z,idx_nom.x[t+2])
        # u_nom = view(z,idx_nom.u[t][u_policy])
        # h_nom = view(z,idx_nom.h[t])
        #
        # K = view(z,idx_K[t])

        for i = 1:N
            # x1i = view(z,idx_sample[i].x[t])
            # x2i = view(z,idx_sample[i].x[t+1])
            # x3i = view(z,idx_sample[i].x[t+2])
            # ui = view(z,idx_sample[i].u[t][u_policy])
            # ūi = view(z,idx_sample[i].u[t][(nu_policy+1):nu])
            # hi = z[idx_sample[i].h[t]]
            #
            # pK(y) = policy(prob.models[i],y,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            # px1i(y) = policy(prob.models[i],K,y,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            # px2i(y) = policy(prob.models[i],K,x1i,y,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            # px3i(y) = policy(prob.models[i],K,x1i,x2i,y,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            # pūi(y) = policy(prob.models[i],K,x1i,x2i,x3i,y,hi,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            # phi(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,y,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
            # px1_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,y,x2_nom,x3_nom,u_nom,h_nom)
            # px2_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,y,x3_nom,u_nom,h_nom)
            # px3_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,y,u_nom,h_nom)
            # pu_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,y,h_nom)
            # ph_nom(y) = policy(prob.models[i],K,x1i,x2i,x3i,ūi,hi,x1_nom,x2_nom,x3_nom,u_nom,y)

            r_idx = r_shift + shift .+ (1:nu_policy)

            c_idx = idx_K[t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pK,K))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1i,x1i))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t+1]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2i,x2i))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].x[t+2]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3i,x3i))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t][(nu_policy+1):nu]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pūi,ūi))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].h[t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(phi,view(z,idx_sample[i].h[t]])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.x[t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1_nom,x1_nom))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.x[t+1]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2_nom,x2_nom))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.x[t+2]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3_nom,x3_nom))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.u[t][u_policy]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pu_nom,u_nom))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_nom.h[t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(phi,view(z,idx_nom.h[t]])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_sample[i].u[t][u_policy]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(-Im)
            # s += len
            row_col!(row,col,r_idx,c_idx)

            shift += nu_policy
        end
    end

    return collect(zip(row,col))
end
