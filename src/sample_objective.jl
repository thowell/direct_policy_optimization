function sample_objective(z,prob::DPOProblem)
    β = prob.β_resample
    N = prob.N_sample_con
    sample_model = prob.sample_model

    J = 0.0

    for t = 1:T-1
        x_nom = view(z,prob.prob.idx.x[t])
        u_nom = view(z,prob.prob.idx.u[t])
        μ = view(z,prob.idx_μ[t])
        L = view(z,prob.idx_L[t])
        K = view(z,prob.idx_K[t])

        Q = prob.Q[t]
        R = prob.R[t]

        J += sample_cost(x_nom,u_nom,μ,L,K,β,N,sample_model,Q,R)
    end

    x_nom = view(z,prob.prob.idx.x[T])
    μ = view(z,prob.idx_μ[T])
    L = view(z,prob.idx_L[T])

    Q = prob.Q[T]
    J += sample_cost_terminal(x_nom,μ,L,β,N,sample_model,Q)

    return J/N
end

function ∇sample_objective!(∇obj,z,prob::DPOProblem)
    β = prob.β_resample
    N = prob.N_sample_con
    sample_model = prob.sample_model

    J = 0.0

    for t = 1:T-1
        x_nom = view(z,prob.prob.idx.x[t])
        u_nom = view(z,prob.prob.idx.u[t])
        μ = view(z,prob.idx_μ[t])
        L = view(z,prob.idx_L[t])
        K = view(z,prob.idx_K[t])
        Q = prob.Q[t]
        R = prob.R[t]

        tmp_x_nom(y) = sample_cost(y,u_nom,μ,L,K,β,N,sample_model,Q,R)
        tmp_u_nom(y) = sample_cost(x_nom,y,μ,L,K,β,N,sample_model,Q,R)
        tmp_μ(y) = sample_cost(x_nom,u_nom,y,L,K,β,N,sample_model,Q,R)
        tmp_L(y) = sample_cost(x_nom,u_nom,μ,y,K,β,N,sample_model,Q,R)
        tmp_K(y) = sample_cost(x_nom,u_nom,μ,L,y,β,N,sample_model,Q,R)

        ∇obj[prob.prob.idx.x[t]] += ForwardDiff.gradient(tmp_x_nom,x_nom)./N
        ∇obj[prob.prob.idx.u[t]] += ForwardDiff.gradient(tmp_u_nom,u_nom)./N
        ∇obj[prob.idx_μ[t]] += ForwardDiff.gradient(tmp_μ,μ)./N
        ∇obj[prob.idx_L[t]] += ForwardDiff.gradient(tmp_L,L)./N
        ∇obj[prob.idx_K[t]] += ForwardDiff.gradient(tmp_K,K)./N
    end

    x_nom = view(z,prob.prob.idx.x[T])
    μ = view(z,prob.idx_μ[T])
    L = view(z,prob.idx_L[T])
    Q = prob.Q[T]

    Tmp_x_nom(y) = sample_cost_terminal(y,μ,L,β,N,sample_model,Q)
    Tmp_μ(y) = sample_cost_terminal(x_nom,y,L,β,N,sample_model,Q)
    Tmp_L(y) = sample_cost_terminal(x_nom,μ,y,β,N,sample_model,Q)

    ∇obj[prob.prob.idx.x[T]] += ForwardDiff.gradient(Tmp_x_nom,x_nom)./N
    ∇obj[prob.idx_μ[T]] += ForwardDiff.gradient(Tmp_μ,μ)./N
    ∇obj[prob.idx_L[T]] += ForwardDiff.gradient(Tmp_L,L)./N

    return nothing
end
