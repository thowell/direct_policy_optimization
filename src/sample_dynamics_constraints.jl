function sample_dynamics_constraints!(c,z,prob::DPOProblem)
    sample_model = prob.sample_model
    β = prob.β_resample
    W = prob.W
    T = prob.prob.T
    Δt = prob.prob.Δt
    N = prob.N_sample_dyn
    n = sample_model.nx
    shift = 0

    # dynamics + resampling (μ1 and L1 are taken care of w/ primal bounds)
    for t = 1:T-1
        x_nom = z[prob.prob.idx.x[t]]
        u_nom = z[prob.prob.idx.u[t]]

        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]
        K = z[prob.idx_K[t]]

        μ⁺ = z[prob.idx_μ[t+1]]
        L⁺ = z[prob.idx_L[t+1]]

        c[(t-1)*(n + n_tri(n)) .+ (1:n)] = (dynamics_sample_mean(x_nom,u_nom,μ,L,K,W[t],β,Δt,t,sample_model,N)
            - μ⁺)

        c[(t-1)*(n + n_tri(n)) + n .+ (1:n_tri(n))] = (dynamics_sample_L(x_nom,model.nu==1 ? u_nom[1] : u_nom,μ,L,K,W[t],β,Δt,t,sample_model,N)
            - L⁺)
    end
    nothing
end

function ∇sample_dynamics_constraints!(∇c,z,prob::DPOProblem)
    sample_model = prob.sample_model
    β = prob.β_resample
    W = prob.W
    T = prob.prob.T
    Δt = prob.prob.Δt
    N = prob.N_sample_dyn
    n = sample_model.nx

    n = prob.prob.model.nx
    shift = 0

    # dynamics + resampling (μ1 and L1 are taken care of w/ primal bounds)
    for t = 1:T-1
        x_nom = z[prob.prob.idx.x[t]]
        u_nom = z[prob.prob.idx.u[t]]

        μ = z[prob.idx_μ[t]]
        L = z[prob.idx_L[t]]
        K = z[prob.idx_K[t]]
        μ⁺ = z[prob.idx_μ[t+1]]
        L⁺ = z[prob.idx_L[t+1]]

        # μ
        tmp1_x_nom(y) = dynamics_sample_mean(y,u_nom,μ,L,K,W[t],β,Δt,t,sample_model,N) - μ⁺
        tmp1_u_nom(y) = dynamics_sample_mean(x_nom,y,μ,L,K,W[t],β,Δt,t,sample_model,N) - μ⁺
        tmp1_μ(y) = dynamics_sample_mean(x_nom,u_nom,y,L,K,W[t],β,Δt,t,sample_model,N) - μ⁺
        tmp1_L(y) = dynamics_sample_mean(x_nom,u_nom,μ,y,K,W[t],β,Δt,t,sample_model,N) - μ⁺
        tmp1_K(y) = dynamics_sample_mean(x_nom,u_nom,μ,L,y,W[t],β,Δt,t,sample_model,N) - μ⁺
        tmp1_μ⁺(y) = dynamics_sample_mean(x_nom,u_nom,μ,L,K,W[t],β,Δt,t,sample_model,N) - y

        r_idx = (t-1)*(n + n_tri(n)) .+ (1:n)
        c_idx = prob.prob.idx.x[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp1_x_nom,x_nom))
        shift += len

        c_idx = prob.prob.idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp1_u_nom,u_nom))
        shift += len

        c_idx = prob.idx_μ[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp1_μ,μ))
        shift += len

        c_idx = prob.idx_L[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp1_L,L))
        shift += len

        c_idx = prob.idx_K[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp1_K,K))
        shift += len

        c_idx = prob.idx_μ[t+1]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp1_μ⁺,μ⁺))
        shift += len

        # L
        tmp2_x_nom(y) = dynamics_sample_L(y,u_nom,μ,L,K,W[t],β,Δt,t,sample_model,N) - L⁺
        tmp2_u_nom(y) = dynamics_sample_L(x_nom,y,μ,L,K,W[t],β,Δt,t,sample_model,N) - L⁺
        tmp2_μ(y) = dynamics_sample_L(x_nom,u_nom,y,L,K,W[t],β,Δt,t,sample_model,N) - L⁺
        tmp2_L(y) = dynamics_sample_L(x_nom,u_nom,μ,y,K,W[t],β,Δt,t,sample_model,N) - L⁺
        tmp2_K(y) = dynamics_sample_L(x_nom,u_nom,μ,L,y,W[t],β,Δt,t,sample_model,N) - L⁺
        tmp2_L⁺(y) = dynamics_sample_L(x_nom,u_nom,μ,L,K,W[t],β,Δt,t,sample_model,N) - y

        r_idx = (t-1)*(n + n_tri(n)) + n .+ (1:n_tri(n))
        c_idx = prob.prob.idx.x[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(real.(FiniteDiff.finite_difference_jacobian(tmp2_x_nom,x_nom)))
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp2_x_nom,x_nom))

        shift += len

        c_idx = prob.prob.idx.u[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = (model.nu==1
            ? vec(real.(FiniteDiff.finite_difference_derivative(tmp2_u_nom,u_nom[1])))
            : vec(real.(FiniteDiff.finite_difference_jacobian(tmp2_u_nom,u_nom)))
            )
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp2_u_nom,u_nom))
        shift += len

        c_idx = prob.idx_μ[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(real.(FiniteDiff.finite_difference_jacobian(tmp2_μ,μ)))
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp2_μ,μ))
        shift += len

        c_idx = prob.idx_L[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(real.(FiniteDiff.finite_difference_jacobian(tmp2_L,L)))
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp2_L,L))
        shift += len

        c_idx = prob.idx_K[t]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(real.(FiniteDiff.finite_difference_jacobian(tmp2_K,K)))
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp2_K,K))
        shift += len

        c_idx = prob.idx_L[t+1]
        len = length(r_idx)*length(c_idx)
        ∇c[shift .+ (1:len)] = vec(real.(FiniteDiff.finite_difference_jacobian(tmp2_L⁺,L⁺)))
        # ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(tmp2_L⁺,L⁺))
        shift += len
    end
    nothing
end

function sparsity_jacobian_sample_dynamics(prob::DPOProblem;
        r_shift=0)

    β = prob.β_resample
    W = prob.W
    T = prob.prob.T
    N = prob.N_sample_dyn
    n = prob.prob.model.nx

    row = []
    col = []

    for t = 1:T-1

        r_idx = r_shift + (t-1)*(n + n_tri(n)) .+ (1:n)
        c_idx = prob.prob.idx.x[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.prob.idx.u[t]
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
        r_idx = r_shift + (t-1)*(n + n_tri(n)) + n .+ (1:n_tri(n))
        c_idx = prob.prob.idx.x[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.prob.idx.u[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_μ[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_L[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_K[t]
        row_col!(row,col,r_idx,c_idx)

        c_idx = prob.idx_L[t+1]
        row_col!(row,col,r_idx,c_idx)
    end
    return collect(zip(row,col))
end
