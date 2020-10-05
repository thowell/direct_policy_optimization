function sample_dynamics_constraints!(c,z,prob::DPOProblem)
    models = prob.models
    β = prob.β_resample
    W = prob.W
    T = prob.prob.T
    N = prob.N

    shift = 0

    # dynamics + resampling (μ1 and L1 are taken care of w/ primal bounds)
    for t = 1:T-1
        x_nom = z[prob.prob.idx.x[t]]
        u_nom = z[prob.prob.idx.u[t]]

        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]
        K = z[prob.idx_K[t]]

        c[(t-1)*(n + n_tri(n)) .+ (1:n)] = (dynamics_sample_mean(x_nom,u_nom,μ,L,K,W[t],β,Δt,t,models,N)
            - z[prob.idx_μ[t+1]])

        c[(t-1)*(n + n_tri(n)) + n .+ (1:n_tri(n))] = (dynamics_sample_L(x_nom,u_nom,μ,L,K,u_vec,W[t],β,Δt,t,models,N)
            - z[prob.idx_L[t+1]])
    end
    nothing
end

function ∇sample_dynamics_constraints!(∇c,z,prob::DPOProblem)
    models = prob.models
    β = prob.β_resample
    W = prob.W
    T = prob.prob.T
    N = prob.N

    n = prob.prob.model.nx
    shift = 0

    # dynamics + resampling (μ1 and L1 are taken care of w/ primal bounds)
    for t = 1:T-1
        x_nom = z[prob.prob.idx.x[t]]
        u_nom = z[prob.prob.idx.u[t]]

        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]
        K = z[prob.idx_K[t]]

        # μ
        tmp1_x_nom(y) = dynamics_sample_mean(y,u_nom,μ,L,K,W[t],β,Δt,t,models,N)
        tmp1_u_nom(y) = dynamics_sample_mean(x_nom,y,μ,L,K,W[t],β,Δt,t,models,N)
        tmp1_μ(y) = dynamics_sample_mean(x_nom,u_nom,y,L,K,W[t],β,Δt,t,models,N)
        tmp1_L(y) = dynamics_sample_mean(x_nom,u_nom,μ,y,K,W[t],β,Δt,t,models,N)
        tmp1_K(y) = dynamics_sample_mean(x_nom,u_nom,μ,L,y,W[t],β,Δt,t,models,N)
        tmp1_μ⁺(y) = dynamics_sample_mean(x_nom,u_nom,μ,L,K,W[t],β,Δt,t,models,N) - y

        r_idx = (t-1)*(n + n_tri(n)) .+ (1:n)
        c_idx = prob.idx_x_nom[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.jacobian(tmp1_x_nom,x_nom)
        shift += len

        c_idx = prob.idx_u_nom[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.jacobian(tmp1_u_nom,u_nom)
        shift += len

        c_idx = prob.idx_μ[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.jacobian(tmp1_μ,μ)
        shift += len

        c_idx = prob.idx_L[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.jacobian(tmp1_L,L)
        shift += len

        c_idx = prob.idx_K[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.jacobian(tmp1_K,K)
        shift += len

        c_idx = prob.idx_μ[t+1]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = ForwardDiff.jacobian(tmp1_μ⁺,μ⁺)
        shift += len

        # L
        tmp2_x_nom(y) = dynamics_sample_covariance(y,u_nom,μ,L,K,W[t],β,Δt,t,models,N)
        tmp2_u_nom(y) = dynamics_sample_covariance(x_nom,y,μ,L,K,W[t],β,Δt,t,models,N)
        tmp2_μ(y) = dynamics_sample_covariance(x_nom,u_nom,y,L,K,W[t],β,Δt,t,models,N)
        tmp2_L(y) = dynamics_sample_covariance(x_nom,u_nom,μ,y,K,W[t],β,Δt,t,models,N)
        tmp2_K(y) = dynamics_sample_covariance(x_nom,u_nom,μ,L,y,W[t],β,Δt,t,models,N)
        tmp2_L⁺(y) = dynamics_sample_covariance(x_nom,u_nom,μ,L,K,W[t],β,Δt,t,models,N) - y

        r_idx = (t-1)*(n + n_tri(n)) .+ (1:n)
        c_idx = prob.idx_x_nom[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec.(real.(FiniteDiff.finite_difference_jacobian(tmp2_x_nom,x_nom)))
        shift += len

        c_idx = prob.idx_u_nom[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec.(real.(ForwardDiff.jacobian(tmp2_u_nom,u_nom)))
        shift += len

        c_idx = prob.idx_μ[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec.(real.(ForwardDiff.jacobian(tmp2_μ,μ)))
        shift += len

        c_idx = prob.idx_L[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec.(real.(ForwardDiff.jacobian(tmp2_L,L)))
        shift += len

        c_idx = prob.idx_K[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec.(real.(ForwardDiff.jacobian(tmp2_K,K)))
        shift += len

        c_idx = prob.idx_μ[t+1]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec.(real.(ForwardDiff.jacobian(tmp2_L⁺,L⁺)))
        shift += len
    end
    nothing
end

function sparsity_jacobian_sample_dynamics(prob::DPOProblem;
        r_shift=0)

    models = prob.models
    β = prob.β_resample
    W = prob.W
    T = prob.prob.T
    N = prob.N
    n = prob.prob.model.nx

    shift = 0

    row = []
    col = []

    for t = 1:T-1

        r_idx = shift + (t-1)*(n + n_tri(n)) .+ (1:n)
        c_idx = prob.idx_x_nom[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_u_nom[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_μ[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_L[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_K[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_μ[t+1]
        row_col!(row,col,r_idx,c_idx)

        # L
        r_idx = shift + (t-1)*(n + n_tri(n)) .+ (1:n)
        c_idx = prob.idx_x_nom[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_u_nom[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_μ[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_L[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_K[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_μ[t+1]
        row_col!(row,col,r_idx,c_idx)
    end
    return collect(zip(row,col))
end
