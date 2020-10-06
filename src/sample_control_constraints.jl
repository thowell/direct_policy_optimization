function sample_control_bounds!(c,z,prob::DPOProblem)
    sample_model = prob.sample_model
    β = prob.β_con
    T = prob.prob.T
    N = prob.N_sample_con
    n = sample_model.nx
    shift = 0

    for t = 1:T-1
        x_nom = z[prob.prob.idx.x[t]]
        u_nom = z[prob.prob.idx.u[t]]

        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]
        K = z[prob.idx_K[t]]

        sample_control_bounds!(view(c,
            (t-1)*(N*2*sample_model.nu) .+ (1:N*2*sample_model.nu)),
            μ,L,K,x_nom,u_nom,β,N,sample_model,prob.ul[t],prob.uu[t])
    end
    nothing
end

function ∇sample_control_bounds!(∇c,z,prob::DPOProblem)
    sample_model = prob.sample_model
    β = prob.β_con
    T = prob.prob.T
    N = prob.N_sample_con
    n = sample_model.nx
    shift = 0
    c_tmp = zeros(N*2*sample_model.nu)

    for t = 1:T-1
        x_nom = z[prob.prob.idx.x[t]]
        u_nom = z[prob.prob.idx.u[t]]

        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]
        K = z[prob.idx_K[t]]

        tmp_μ(c,y) = sample_control_bounds!(c,y,L,K,x_nom,u_nom,β,N,
            sample_model,prob.ul[t],prob.uu[t])
        tmp_L(c,y) = sample_control_bounds!(c,μ,y,K,x_nom,u_nom,β,N,
            sample_model,prob.ul[t],prob.uu[t])
        tmp_K(c,y) = sample_control_bounds!(c,μ,L,y,x_nom,u_nom,β,N,
            sample_model,prob.ul[t],prob.uu[t])
        tmp_x_nom(c,y) = sample_control_bounds!(c,μ,L,K,y,u_nom,β,N,
            sample_model,prob.ul[t],prob.uu[t])
        tmp_u_nom(c,y) = sample_control_bounds!(c,μ,L,K,x_nom,y,β,N,
            sample_model,prob.ul[t],prob.uu[t])

        r_idx = (t-1)*(N*2*sample_model.nu) .+ (1:N*2*sample_model.nu)

        c_idx = prob.idx_μ[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_μ,c_tmp,μ))
        shift += len

        c_idx = prob.idx_L[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_L,c_tmp,L))
        shift += len

        c_idx = prob.idx_K[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_K,c_tmp,K))
        shift += len

        c_idx = prob.prob.idx.x[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_x_nom,c_tmp,x_nom))
        shift += len

        c_idx = prob.prob.idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_u_nom,c_tmp,u_nom))
        shift += len
    end
    nothing
end

function sparsity_jacobian_sample_control_bounds(prob::DPOProblem;
        r_shift=0)

    sample_model = prob.sample_model
    β = prob.β_con
    T = prob.prob.T
    N = prob.N_sample_con
    n = sample_model.nx
    shift = 0
    c_tmp = zeros(N*2*sample_model.nu)

    row = []
    col = []

    if prob.M_control > 0
        for t = 1:T-1
            r_idx = r_shift + (t-1)*(N*2*sample_model.nu) .+ (1:N*2*sample_model.nu)

            c_idx = prob.idx_μ[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = prob.idx_L[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = prob.idx_K[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = prob.prob.idx.x[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = prob.prob.idx.u[t]
            row_col!(row,col,r_idx,c_idx)
        end
    end
    return collect(zip(row,col))
end
