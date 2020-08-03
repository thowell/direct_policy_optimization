mutable struct SampleProblem <: Problem
    prob::TrajectoryOptimizationProblem

    M_sample::Int # number of sample constraints
    N_nlp::Int # number of decision variables
    M_nlp::Int # number of constraints

    idx_nom
    idx_nom_z
    idx_sample
    idx_x_tmp
    idx_K

    Q
    R

    N::Int # number of samples
    models
    x1
    β
    w
    γ
end

function init_sample_problem(prob::TrajectoryOptimizationProblem,models,x1,Q,R;
        time=true,β=1.0,w=1.0,γ=1.0)

    nx = prob.n
    nu = prob.m
    T = prob.T
    N = length(models)
    @assert N == 2*nx

    M_sample = N*2*nx*(T-1) + N*nu*(T-1) + N*prob.m_stage*(T-1)
    N_nlp = prob.N + N*(nx*T + nu*(T-1)) + N*(nx*(T-1)) + nu*nx*(T-1)
    M_nlp = prob.M + M_sample

    idx_nom = init_indices(nx,nu,T,time=time,shift=0)
    idx_nom_z = 1:prob.N
    shift = nx*T + nu*(T-1) + (T-1)*true
    idx_sample = [init_indices(nx,nu,T,time=false,shift=shift + (i-1)*(nx*T + nu*(T-1))) for i = 1:N]
    shift += N*(nx*T + nu*(T-1))
    idx_x_tmp = [init_indices(nx,0,T-1,time=false,shift=shift + (i-1)*(nx*(T-1))) for i = 1:N]
    shift += N*(nx*(T-1))
    idx_K = [shift + (t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]

    return SampleProblem(prob,
        M_sample,N_nlp,M_nlp,
        idx_nom,idx_nom_z,
        idx_sample,idx_x_tmp,idx_K,
        Q,R,
        N,models,x1,β,w,γ)
end

function pack(X0,U0,h0,K0,prob::SampleProblem)
    Z0 = zeros(prob.N_nlp)
    Z0[prob.idx_nom_z] = pack(X0,U0,h0,prob.prob)
    T = prob.prob.T
    N = prob.N

    for t = 1:T
        for i = 1:N
            Z0[prob.idx_sample[i].x[t]] = X0[t]
            t==T && continue
            Z0[prob.idx_sample[i].u[t]] = U0[t]
            Z0[prob.idx_x_tmp[i].x[t]] = X0[t+1]
        end
    end

    for t = 1:T-1
        Z0[prob.idx_K[t]] = K0[t]
    end
    return Z0
end

function unpack(Z0,prob::SampleProblem)
    T = prob.prob.T
    N = prob.N

    X_nom = [Z0[prob.idx_nom.x[t]] for t = 1:T]
    U_nom = [Z0[prob.idx_nom.u[t]] for t = 1:T-1]
    H_nom = [Z0[prob.idx_nom.h[t]] for t = 1:T-1]

    X_sample = [[Z0[prob.idx_sample[i].x[t]] for t = 1:T] for i = 1:N]
    U_sample = [[Z0[prob.idx_sample[i].u[t]] for t = 1:T-1] for i = 1:N]

    return X_nom, U_nom, H_nom, X_sample, U_sample
end

function init_MOI_Problem(prob::SampleProblem)
    return MOIProblem(prob.N_nlp,prob.M_nlp,prob,false)
end


function primal_bounds(prob::SampleProblem)
    Zl = -Inf*ones(prob.N_nlp)
    Zu = Inf*ones(prob.N_nlp)

    # nominal bounds
    Zl_nom, Zu_nom = primal_bounds(prob.prob)
    Zl[prob.idx_nom_z] = Zl_nom
    Zu[prob.idx_nom_z] = Zu_nom

    # sample initial conditions
    for i = 1:prob.N
        Zl[prob.idx_sample[i].x[1]] = prob.x1[i]
        Zu[prob.idx_sample[i].x[1]] = prob.x1[i]
    end

    # sample state and control bounds
    for t = 1:prob.prob.T-1
        for i = 1:prob.N
            Zl[prob.idx_sample[i].u[t]] = prob.prob.ul[t]
            Zu[prob.idx_sample[i].u[t]] = prob.prob.uu[t]

            t==1 && continue
            Zl[prob.idx_sample[i].x[t]] = prob.prob.xl[t]
            Zu[prob.idx_sample[i].x[t]] = prob.prob.xu[t]
        end
    end

    #TODO sample goal constraints

    return Zl,Zu
end

function constraint_bounds(prob::SampleProblem)
    cl = zeros(prob.M_nlp)
    cu = zeros(prob.M_nlp)

    # nominal constraints
    M_nom = prob.prob.M
    cl_nom, cu_nom = constraint_bounds(prob.prob)

    cl[1:M_nom] = cl_nom
    cu[1:M_nom] = cu_nom

    # sample stage constraints
    if prob.prob.m_stage > 0
        cu[M_nom+prob.N*2*prob.prob.n*(prob.prob.T-1) + prob.N*prob.prob.m*(prob.prob.T-1) .+ (1:prob.N*prob.prob.m_stage*(prob.prob.T-1))] .= Inf
    end
    return cl,cu
end

function eval_objective(prob::SampleProblem,Z)
    (eval_objective(prob.prob,view(Z,prob.idx_nom_z))
        + obj_sample(Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.prob.T,
            prob.N,prob.γ))
end

function eval_objective_gradient!(∇obj,Z,prob::SampleProblem)
    ∇obj .= 0.0
    eval_objective_gradient!(view(∇obj,prob.idx_nom_z),view(Z,prob.idx_nom_z),
        prob.prob)
    ∇obj_sample!(∇obj,Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.prob.T,prob.N,prob.γ)
    return nothing
end

function eval_constraint!(c,Z,prob::SampleProblem)
   M_nom = prob.prob.M
   M_sample = prob.M_sample
   eval_constraint!(view(c,1:M_nom),view(Z,prob.idx_nom_z),prob.prob)
   con_sample!(view(c,M_nom .+ (1:M_sample)),Z,prob.idx_nom,prob.idx_sample,prob.idx_x_tmp,
        prob.idx_K,prob.Q,prob.R,prob.models,prob.β,prob.w,
        prob.prob.m_stage,prob.prob.T,prob.N,prob.prob.integration)
   return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::SampleProblem)
    len = length(sparsity_jacobian(prob.prob))
    eval_constraint_jacobian!(view(∇c,1:len),view(Z,prob.idx_nom_z),prob.prob)

    M_nom = prob.prob.M
    M_sample = prob.M_sample

    len_sample = length(sparsity_jacobian_sample(prob.idx_nom,
        prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.prob.m_stage,prob.prob.T,
        prob.N))

    ∇con_sample_vec!(view(∇c,len .+ (1:len_sample)),
         Z,prob.idx_nom,
         prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.Q,prob.R,prob.models,
         prob.β,prob.w,prob.prob.m_stage,prob.prob.T,prob.N,
         prob.prob.integration)

    # con_tmp(c,z) = con_sample!(c,z,prob.idx_nom,prob.idx_sample,prob.idx_x_tmp,
    #      prob.idx_K,prob.Q,prob.R,prob.models,prob.β,prob.w,prob.prob.con,
    #      prob.prob.m_stage,prob.prob.T,prob.N,prob.prob.integration)
    #
    # ∇c[len .+ (1:M_sample*prob.N_nlp)] = vec(ForwardDiff.jacobian(con_tmp,zeros(M_sample),Z))
    return nothing
end

function sparsity_jacobian(prob::SampleProblem)
    M_nom = prob.prob.M
    M_sample = prob.M_sample
    collect([sparsity_jacobian(prob.prob)...,
        sparsity_jacobian_sample(prob.idx_nom,
        prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.prob.m_stage,
        prob.prob.T,prob.N,r_shift=M_nom)...])
end
