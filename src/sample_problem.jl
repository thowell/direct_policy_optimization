mutable struct DPOProblem <: Problem
    prob::TrajectoryOptimizationProblem

    u_policy
    n_policy
    n_features

    N_nlp::Int # number of decision variables
    Nμ::Int
    NP::Int
    NK::Int
    Nu::Int

    M_nlp::Int # number of constraints
    M_dynamics::Int # number of sample constraints
    M_time::Int
    M_policy::Int
    M_stage::Int
    M_general::Int

    idx_μ
    idx_P
    idx_K
    idx_u

    Q
    R

    μl
    μu

    Pl
    Pu

    Kl
    Ku

    ul
    uu

    N::Int # number of samples
    models
    β_resample
    β_con
    W

    general_objective

    sample_general_constraints
    m_sample_general
    sample_general_ineq
end

function init_DPO_problem(prob::TrajectoryOptimizationProblem,models,Q,R;
        n_policy=prob.nu,
        n_features=prob.nx,
        μl=[-Inf*ones(models[1].nx) for t = 1:prob.T],
        μu=[Inf*ones(models[1].nx) for t = 1:prob.T],
        Ll=[-Inf*ones(n_tri(models[1].nx)) for t = 1:prob.T],
        Lu=[Inf*ones(n_tri(models[1].nx)) for t = 1:prob.T],
        Kl=[-Inf*ones(n_policy*n_features) for t = 1:T-1],
        Ku=[Inf*ones(n_policy*n_features) for t = 1:T-1],
        ul=[prob.ul for i = 1:length(models)],
        uu=[prob.uu for i = 1:length(models)],
        β_resample=1.0,β_con=1.0,W=[ones(prob.nx) for t = 1:T-1],
        general_objective=false,
        sample_general_constraints=false,
        m_sample_general=0,
        sample_general_ineq=(1:m_sample_general)
        )

    u_policy = 1:n_policy

    T = prob.T
    N = length(models)

    Nμ = sum([models[i].nx for i = 1:N])*T
    NL = sum([n_tri(models[i].nx) for i = 1:N])*T
    NK = n_policy*n_features*(T-1)
    Nu = sum([models[i].nu for i = 1:N])*(T-1)

    N_nlp = prob.N + Nμ + NL + NK + Nu

    M_dynamics = sum([models[i].nx + n_tri(models[i].nx) for i = 1:N])*(T-1)
    M_time = prob.free_time*N*(T-2)
    M_policy = policy_constraint*N*n_policy*(T-1)
    M_stage = prob.stage_constraints*N*sum(prob.m_stage)
    M_general = sample_general_constraints*m_sample_general

    M_nlp = prob.M + M_dynamics + M_time + M_policy + M_stage + M_general

    shift = prob.N

    idx_μ = [shift + (t-1)*models[i].nx .+ (1:model[i].nx) for t = 1:T]
    shift += sum([models[i].nx for i = 1:N])*T

    idx_L = [shift + (t-1)*n_tri(models[i].nx) .+ (1:n_tri(model[i].nx)) for t = 1:T]
    shift += sum([n_tri(models[i].nx) for i = 1:N])*T

    idx_K = [shift + (t-1)*(n_policy*n_features) .+ (1:n_policy*n_features) for t = 1:T-1]
    shift += (T-1)*n_policy*n_features

    idx_u = [[shift (i-1)*(i==1 ? 0 : models[i-1].nu) + (t-1)*models[i].nu .+ (1:models[i].nu)] for i = 1:N]

    return DPOProblem(
        prob,
        u_policy,
        n_policy,n_features,
        N_nlp,Nμ,NL,NK,Nu,
        M_nlp,M_dynamics,M_time,M_policy,M_stage,M_general,
        idx_μ,idx_P,idx_K,idx_U,
        Q,R,
        μl,μu,
        Ll,Lu,
        Kl,Ku,
        ul,uu,
        N,models,β,w,
        general_objective,
        policy_constraint,
        sample_general_constraints,
        m_sample_general,
        sample_general_ineq
        )
end

function pack(X0,U0,K0,prob::DPOProblem;
        r=0.0)

    model = prob.prob.model

    Z0 = zeros(prob.N_nlp)
    Z0[1:prob.prob.N] = pack(X0,U0,prob.prob)

    T = prob.prob.T
    N = prob.N

    for t = 1:T
       Z0[prob.idx_μ[t]] = copy(X0[t]) + r*rand(model.nx)
       Z0[prob.idx_L[t]] .= 0.0
    end

    for t = 1:T-1
        Z0[prob.idx_K[t]] = vec(K0[t])

        for i = 1:prob.N
            Z0[prob.idx_u[i][t]] = copy(U0)
        end
    end


    return Z0
end

function unpack(Z0,prob::DPOProblem)
    T = prob.prob.T
    N = prob.N

    X_nom = [Z0[prob.prob.idx.x[t]] for t = 1:T]
    U_nom = [Z0[prob.prob.idx.u[t]] for t = 1:T-1]

    return X_nom, U_nom
end

function init_MOI_Problem(prob::DPOProblem)
    return MOIProblem(prob.N_nlp,prob.M_nlp,prob,
        primal_bounds(prob),constraint_bounds(prob),false)
end


function primal_bounds(prob::DPOProblem)
    Zl = -Inf*ones(prob.N_nlp)
    Zu = Inf*ones(prob.N_nlp)

    # nominal bounds
    Zl_nom, Zu_nom = primal_bounds(prob.prob)
    Zl[1:prob.prob.N] = Zl_nom
    Zu[1:prob.prob.N] = Zu_nom

    # sample state and control bounds
    for t = 1:prob.prob.T
        Zl[prob.idx_μ[t]] = prob.μl[t]
        Zu[prob.idx_μ[t]] = prob.μu[t]

        Zl[prob.idx_L[t]] = prob.Ll[t]
        Zu[prob.idx_L[t]] = prob.Lu[t]

        t == T && continue
        Zl[prob.idx_K[t]] = prob.Kl[t]
        Zu[prob.idx_K[t]] = prob.Ku[t]

        for i = 1:prob.N
            Zl[prob.idx_u[i][t]] = prob.ul[i][t]
            Zu[prob.idx_u[i][t]] = prob.uu[i][t]
        end
    end

    return Zl,Zu
end

function constraint_bounds(prob::DPOProblem)
    cl = zeros(prob.M_nlp)
    cu = zeros(prob.M_nlp)

    # nominal constraints
    M_nom = prob.prob.M
    cl_nom, cu_nom = constraint_bounds(prob.prob)

    cl[1:M_nom] = cl_nom
    cu[1:M_nom] = cu_nom

    # sample stage constraints
    if prob.prob.stage_constraints
        m_shift = 0
        for (t,m_stage) in enumerate(prob.prob.m_stage)
            for i = 1:prob.N
                cu[(M_nom
                    + prob.M_dynamics
                    + prob.M_time
                    + prob.M_policy
                    + m_shift .+ (1:m_stage))[prob.prob.stage_ineq[t]]] .= Inf
                m_shift += m_stage
            end
        end
    end

    prob.sample_general_constraints && (cu[(M_nom
        + prob.M_dynamics
        + prob.M_time
        + prob.M_policy
        + prob.M_stage
        .+ (1:prob.M_general))[prob.sample_general_ineq]] .= Inf)

    return cl,cu
end

function eval_objective(prob::DPOProblem,Z)
    J = 0.0
    J += eval_objective(prob.prob,view(Z,1:prob.prob.N))

    J += sample_objective(Z,prob)

    (prob.general_objective ? (J += sample_general_objective(Z,prob)) : 0.0)

    return J
end

function eval_objective_gradient!(∇obj,Z,prob::DPOProblem)
    ∇obj .= 0.0

    eval_objective_gradient!(view(∇obj,prob.idx_nom_z),view(Z,prob.idx_nom_z),
        prob.prob)

    ∇sample_objective!(∇obj,Z,prob)

    prob.general_objective && ∇sample_general_objective!(∇obj,Z,prob)

    return nothing
end

function eval_constraint!(c,Z,prob::DPOProblem)
   M = prob.prob.M

   eval_constraint!(view(c,1:M),view(Z,1:prob.prob.N),prob.prob)

   sample_dynamics_constraints!(view(c,M .+ (1:prob.M_dynamics)),Z,prob)

   prob.prob.free_time && sample_time_constraints!(view(c,
    M+prob.M_dynamics .+ (1:prob.M_time)),Z,prob)

   prob.policy_constraint && policy_constraints!(view(c,
    M+prob.M_dynamics+prob.M_time .+ (1:prob.M_policy)),Z,prob)

   prob.prob.stage_constraints && sample_stage_constraints!(view(c,
    M+prob.M_dynamics+prob.M_time+prob.M_policy .+ (1:prob.M_stage)),Z,prob)

   prob.sample_general_constraints && general_constraints!(view(c,
    M+prob.M_dynamics+prob.M_time+prob.M_policy+prob.M_stage .+ (1:prob.M_general)),Z,prob)

   return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::DPOProblem)
    len = length(sparsity_jacobian(prob.prob))
    eval_constraint_jacobian!(view(∇c,1:len),view(Z,prob.idx_nom_z),prob.prob)

    len_dyn = length(sparsity_jacobian_sample_dynamics(prob))
    ∇sample_dynamics_constraints!(view(∇c,len .+ (1:len_dyn)),Z,prob)

    len_time = length(sparsity_jacobian_sample_time(prob))
    ∇sample_time_constraints!(view(∇c,len+len_dyn .+ (1:len_time)),Z,prob)

    len_policy = length(sparsity_jacobian_policy(prob))
    prob.policy_constraint && ∇policy_constraints!(view(∇c,
        len+len_dyn+len_time .+ (1:len_policy)),Z,prob)

    len_stage = length(sparsity_jacobian_sample_stage(prob))
    prob.prob.stage_constraints && ∇sample_stage_constraints!(view(∇c,
        len+len_dyn+len_policy+len_time .+ (1:len_stage)),Z,prob)

    len_general = length(general_constraint_sparsity(prob))
    prob.sample_general_constraints && ∇general_constraints!(view(∇c,
        len+len_dyn++len_time+len_policy+len_stage .+ (1:len_general)),Z,prob)
    return nothing
end

function sparsity_jacobian(prob::DPOProblem)
    collect([sparsity_jacobian(prob.prob)...,
             sparsity_jacobian_sample_dynamics(prob,
                r_shift=prob.prob.M)...,
            sparsity_jacobian_sample_time(prob,
               r_shift=prob.prob.M+prob.prob.M_dynamics)...,
             sparsity_jacobian_policy(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_time)...,
             sparsity_jacobian_sample_stage(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_time+prob.M_policy)...,
             general_constraint_sparsity(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_time+prob.M_policy
                +prob.M_stage)...,
             sparsity_jacobian_sample_disturbance(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_time+prob.M_policy
                +prob.M_stage+prob.M_general)...])
end
