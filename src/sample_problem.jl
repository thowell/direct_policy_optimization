mutable struct DPOProblem <: Problem
    prob::TrajectoryOptimizationProblem

    n_policy
    n_features

    N_nlp::Int # number of decision variables
    Nμ::Int
    NP::Int
    NK::Int

    M_nlp::Int # number of constraints
    M_dynamics::Int # number of sample constraints
    M_control::Int
    M_state::Int
    M_stage::Int
    M_general::Int

    idx_μ
    idx_L
    idx_K

    Q
    R

    μl
    μu

    Ll
    Lu

    ul
    uu

    xl
    xu

    N_sample_dyn # samples from joint distribution over x and w
    N_sample_con # samples from distribution over x

    sample_model
    β_resample
    β_con
    W

    general_objective
    sample_control_consraints
    sample_state_constraints
    sample_general_constraints
    m_sample_general
    sample_general_ineq
end

function init_DPO_problem(prob::TrajectoryOptimizationProblem,
        sample_model,
        Q,R;
        n_policy=prob.model.nu,
        n_features=prob.model.nx,
        μl=[-Inf*ones(sample_model.nx) for t = 1:prob.T],
        μu=[Inf*ones(sample_model.nx) for t = 1:prob.T],
        Ll=[-Inf*ones(n_tri(sample_model.nx)) for t = 1:prob.T],
        Lu=[Inf*ones(n_tri(sample_model.nx)) for t = 1:prob.T],
        ul=[-Inf*ones(sample_model.nu) for t = 1:prob.T-1],
        uu=[Inf*ones(sample_model.nu) for t = 1:prob.T-1],
        xl=[-Inf*ones(sample_model.nx) for t = 1:prob.T],
        xu=[Inf*ones(sample_model.nx) for t = 1:prob.T],
        β_resample=1.0,β_con=1.0,W=[Diagonal(ones(sample_model.nx)) for t = 1:T-1],
        general_objective=false,
        sample_control_constraints=false,
        sample_state_constraints=false,
        sample_general_constraints=false,
        m_sample_general=0,
        sample_general_ineq=(1:m_sample_general)
        )

    N_sample_dyn = 2*(sample_model.nx + sample_model.nw)
    N_sample_con = 2*sample_model.nx
    T = prob.T

    Nμ = sample_model.nx*T
    NL = n_tri(sample_model.nx)*T
    NK = n_policy*n_features*(T-1)

    N_nlp = prob.N + Nμ + NL + NK

    M_dynamics = (sample_model.nx + n_tri(sample_model.nx))*(T-1)
    M_control = sample_control_constraints*2*N_sample_con*sample_model.nu*(T-1)
    M_state = sample_state_constraints*2*N_sample_con*sample_model.nx*T
    M_stage = prob.stage_constraints*N_sample_con*sum(prob.m_stage)
    M_general = sample_general_constraints*m_sample_general

    M_nlp = prob.M + M_dynamics + M_control + M_state + M_stage + M_general

    shift = prob.N

    idx_μ = [shift + (t-1)*sample_model.nx .+ (1:sample_model.nx) for t = 1:T]
    shift += sample_model.nx*T

    idx_L = [shift + (t-1)*n_tri(sample_model.nx) .+ (1:n_tri(sample_model.nx)) for t = 1:T]
    shift += n_tri(sample_model.nx)*T

    idx_K = [shift + (t-1)*(n_policy*n_features) .+ (1:n_policy*n_features) for t = 1:T-1]
    shift += (T-1)*n_policy*n_features

    return DPOProblem(
        prob,
        n_policy,n_features,
        N_nlp,Nμ,NL,NK,
        M_nlp,M_dynamics,M_control,M_state,M_stage,M_general,
        idx_μ,idx_L,idx_K,
        Q,R,
        μl,μu,
        Ll,Lu,
        ul,uu,
        xl,xu,
        N_sample_dyn,N_sample_con,sample_model,
        β_resample,β_con,W,
        general_objective,
        sample_control_constraints,
        sample_state_constraints,
        sample_general_constraints,
        m_sample_general,
        sample_general_ineq
        )
end

function pack(X0,U0,M0,L0,K0,prob::DPOProblem;
        r=0.0)

    model = prob.prob.model

    Z0 = zeros(prob.N_nlp)
    Z0[1:prob.prob.N] = pack(X0,U0,prob.prob)

    T = prob.prob.T

    for t = 1:T
       Z0[prob.idx_μ[t]] = copy(M0[t]) + r*rand(prob.sample_model.nx)
       Z0[prob.idx_L[t]] = copy(L0[t]) + r*rand(n_tri(prob.sample_model.nx))
       t==T && continue
       Z0[prob.idx_K[t]] = vec(K0[t])
    end

    return Z0
end

function unpack(Z0,prob::DPOProblem)
    T = prob.prob.T

    X_nom = [Z0[prob.prob.idx.x[t]] for t = 1:T]
    U_nom = [Z0[prob.prob.idx.u[t]] for t = 1:T-1]

    μ = [Z0[prob.idx_μ[t]] for t = 1:T]
    L = [Z0[prob.idx_L[t]] for t = 1:T]
    K = [Z0[prob.idx_K[t]] for t = 1:T-1]

    X_sample = [[μ[t] + s*prob.β_con*vec_to_lt(L[t])[:,i] for t = 1:T] for s in [-1.0,1.0] for i = 1:prob.sample_model.nx]
    U_sample = [[policy(prob.sample_model,K[t],X_sample[i][t],X_nom[t],U_nom[t]) for t = 1:T-1] for i = 1:prob.N_sample_con]

    return X_nom, U_nom, μ, L, K, X_sample, U_sample
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

        # t == T && continue
        # Zl[prob.idx_K[t]] = prob.Kl[t]
        # Zu[prob.idx_K[t]] = prob.Ku[t]
        #
        # for i = 1:prob.N
        #     Zl[prob.idx_u[i][t]] = prob.ul[i][t]
        #     Zu[prob.idx_u[i][t]] = prob.uu[i][t]
        # end
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

    cu[M_nom+prob.M_dynamics .+ (1:prob.M_control)] .= Inf
    cu[M_nom+prob.M_dynamics+prob.M_control .+ (1:prob.M_state)] .= Inf

    # sample stage constraints
    if prob.prob.stage_constraints
        m_shift = 0
        for (t,m_stage) in enumerate(prob.prob.m_stage)
            for i = 1:prob.N_sample_con
                cu[(M_nom
                    + prob.M_dynamics + prob.M_control + prob.M_state
                    + m_shift .+ (1:m_stage))[prob.prob.stage_ineq[t]]] .= Inf
                m_shift += m_stage
            end
        end
    end
    #
    # prob.sample_general_constraints && (cu[(M_nom
    #     + prob.M_dynamics + prob.M_control + prob.M_state
    #     + prob.M_stage
    #     .+ (1:prob.M_general))[prob.sample_general_ineq]] .= Inf)

    return cl,cu
end

function eval_objective(prob::DPOProblem,Z)
    J = 0.0
    J += eval_objective(prob.prob,view(Z,1:prob.prob.N))

    J += sample_objective(Z,prob)
    #
    # (prob.general_objective ? (J += sample_general_objective(Z,prob)) : 0.0)

    return J
end

function eval_objective_gradient!(∇obj,Z,prob::DPOProblem)
    ∇obj .= 0.0

    eval_objective_gradient!(view(∇obj,1:prob.prob.N),view(Z,1:prob.prob.N),
        prob.prob)

    ∇sample_objective!(∇obj,Z,prob)
    #
    # prob.general_objective && ∇sample_general_objective!(∇obj,Z,prob)

    return nothing
end

function eval_constraint!(c,Z,prob::DPOProblem)
   M = prob.prob.M

   eval_constraint!(view(c,1:M),view(Z,1:prob.prob.N),prob.prob)

   sample_dynamics_constraints!(view(c,M .+ (1:prob.M_dynamics)),Z,prob)

   prob.M_control > 0  && sample_control_bounds!(view(c,M+prob.M_dynamics .+ (1:prob.M_control)),Z,prob)

   prob.M_state > 0 && sample_state_bounds!(view(c,M+prob.M_dynamics+prob.M_control .+ (1:prob.M_state)),Z,prob)

   prob.prob.stage_constraints && sample_stage_constraints!(view(c,
    M+prob.M_dynamics+prob.M_control+prob.M_state .+ (1:prob.M_stage)),Z,prob)
   #
   # prob.sample_general_constraints && general_constraints!(view(c,
   #  M+prob.M_dynamics+prob.M_stage .+ (1:prob.M_general)),Z,prob)

   return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::DPOProblem)
    len = length(sparsity_jacobian(prob.prob))
    eval_constraint_jacobian!(view(∇c,1:len),view(Z,1:prob.prob.N),prob.prob)

    len_dyn = length(sparsity_jacobian_sample_dynamics(prob))
    ∇sample_dynamics_constraints!(view(∇c,len .+ (1:len_dyn)),Z,prob)

    len_control = length(sparsity_jacobian_sample_control_bounds(prob))
    prob.M_control > 0 && ∇sample_control_bounds!(view(∇c,len+len_dyn .+ (1:len_control)),Z,prob)

    len_state = length(sparsity_jacobian_sample_state_bounds(prob))
    prob.M_state > 0 && ∇sample_state_bounds!(view(∇c,len+len_dyn+len_control .+ (1:len_state)),Z,prob)

    len_stage = length(sparsity_jacobian_sample_stage(prob))
    prob.prob.stage_constraints && ∇sample_stage_constraints!(view(∇c,
        len+len_dyn+len_control+len_state .+ (1:len_stage)),Z,prob)
    #
    # len_general = length(general_constraint_sparsity(prob))
    # prob.sample_general_constraints && ∇general_constraints!(view(∇c,
    #     len+len_dyn+len_stage .+ (1:len_general)),Z,prob)
    return nothing
end

function sparsity_jacobian(prob::DPOProblem)
    collect([sparsity_jacobian(prob.prob)...,
             sparsity_jacobian_sample_dynamics(prob,
                r_shift=prob.prob.M)...,
             sparsity_jacobian_sample_control_bounds(prob,
                r_shift=prob.prob.M+prob.M_dynamics)...,
             sparsity_jacobian_sample_state_bounds(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_control)...,
             sparsity_jacobian_sample_stage(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_control+prob.M_state)...])#,
             # general_constraint_sparsity(prob,
             #    r_shift=prob.prob.M+prob.M_dynamics+prob.M_stage)...])
end
