function sample_state_bounds!(c,z,prob::DPOProblem)
    sample_model = prob.sample_model
    β = prob.β_con
    T = prob.prob.T
    N = prob.N_sample_con
    n = sample_model.nx
    shift = 0

    for t = 1:T
        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]

        sample_state_bounds!(view(c,
            (t-1)*(N*2*sample_model.nx) .+ (1:N*2*sample_model.nx)),
            μ,L,β,N,sample_model,prob.xl[t],prob.xu[t])
    end
    nothing
end

function ∇sample_state_bounds!(∇c,z,prob::DPOProblem)
    sample_model = prob.sample_model
    β = prob.β_con
    T = prob.prob.T
    N = prob.N_sample_con
    n = sample_model.nx
    shift = 0
    c_tmp = zeros(N*2*sample_model.nx)

    for t = 1:T
        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]

        tmp_μ(c,y) = sample_state_bounds!(c,y,L,β,N,
            sample_model,prob.xl[t],prob.xu[t])
        tmp_L(c,y) = sample_state_bounds!(c,μ,y,β,N,
            sample_model,prob.xl[t],prob.xu[t])

        r_idx = (t-1)*(N*2*sample_model.nx) .+ (1:N*2*sample_model.nx)

        c_idx = prob.idx_μ[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_μ,c_tmp,μ))
        shift += len

        c_idx = prob.idx_L[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp_L,c_tmp,L))
        shift += len
    end
    nothing
end

function sparsity_jacobian_sample_state_bounds(prob::DPOProblem;
        r_shift=0)

    sample_model = prob.sample_model
    β = prob.β_con
    T = prob.prob.T
    N = prob.N_sample_con
    n = sample_model.nx
    shift = 0
    c_tmp = zeros(N*2*sample_model.nx)

    row = []
    col = []

    if prob.M_state > 0
        for t = 1:T
            r_idx = r_shift + (t-1)*(N*2*sample_model.nx) .+ (1:N*2*sample_model.nx)

            c_idx = prob.idx_μ[t]
            row_col!(row,col,r_idx,c_idx)

            c_idx = prob.idx_L[t]
            row_col!(row,col,r_idx,c_idx)
        end
    end
    return collect(zip(row,col))
end
