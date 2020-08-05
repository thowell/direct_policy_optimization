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
    H

    N::Int # number of samples
    models
    x1
    β
    w
    γ

    disturbance_ctrl
    α
    idx_uw
    idx_slack
    M_dist
end

function init_sample_problem(prob::TrajectoryOptimizationProblem,models,x1,Q,R,H;
        time=true,β=1.0,w=1.0,γ=1.0,
        disturbance_ctrl=false,α=1.0)

    nx = prob.n
    nu = prob.m

    T = prob.T
    N = length(models)
    @assert N == 2*nx
    @assert size(R[1],1) == nu

    M_sample = N*(2*nx*(T-1) + (T-2) + nu*(T-1) + prob.stage_constraints*sum(prob.m_stage))
    M_dist = disturbance_ctrl*2*N*nx*(T-1)

    N_nlp = prob.N + N*(nx*T) + N*nu*(T-1) + N*(T-1) + N*(nx*(T-1)) + nu*nx*(T-1) + disturbance_ctrl*2*N*nx*(T-1)
    M_nlp = prob.M + M_sample + M_dist

    idx_nom = init_indices(nx,nu,T,time=time,shift=0)
    idx_nom_z = 1:prob.N
    shift = nx*T + nu*(T-1) + (T-1)*true
    idx_sample = [init_indices(nx,nu,T,time=true,shift=shift + (i-1)*(nx*T + nu*(T-1) + (T-1))) for i = 1:N]
    shift += N*(nx*T) + N*nu*(T-1) + N*(T-1)
    idx_x_tmp = [init_indices(nx,0,T-1,time=false,shift=shift + (i-1)*(nx*(T-1))) for i = 1:N]
    shift += N*(nx*(T-1))
    idx_K = [shift + (t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]
    shift += (T-1)*nu*nx

    if disturbance_ctrl
        idx_uw = [[shift + (i-1)*nx*(T-1) + (t-1)*nx .+ (1:nx)  for t = 1:T-1] for i = 1:N]
        shift += N*nx*(T-1)
        idx_slack = [[shift + (i-1)*nx*(T-1) + (t-1)*nx .+ (1:nx)  for t = 1:T-1] for i = 1:N]
        shift += N*nx*(T-1)
    else
        idx_uw = []
        idx_slack = []
    end

    return SampleProblem(prob,
        M_sample,N_nlp,M_nlp,
        idx_nom,idx_nom_z,
        idx_sample,idx_x_tmp,idx_K,
        Q,R,H,
        N,models,x1,β,w,γ,
        disturbance_ctrl,
        α,
        idx_uw,
        idx_slack,
        M_dist)
end

function pack(X0,U0,h0,K0,prob::SampleProblem;
        uw=0.0,s=1.0)

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

            if prob.disturbance_ctrl
                Z0[prob.idx_uw[i][t]] .= uw
                Z0[prob.idx_slack[i][t]] .= s
            end
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
    H_sample = [[Z0[prob.idx_sample[i].h[t]] for t = 1:T-1] for i = 1:N]

    return X_nom, U_nom, H_nom, X_sample, U_sample, H_sample
end

function init_MOI_Problem(prob::SampleProblem)
    return MOIProblem(prob.N_nlp,prob.M_nlp,prob,
        primal_bounds(prob),constraint_bounds(prob),false)
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

            Zl[prob.idx_sample[i].h[t]] = prob.prob.hl[t]
            Zu[prob.idx_sample[i].h[t]] = prob.prob.hu[t]

            t==1 && continue
            Zl[prob.idx_sample[i].x[t]] = prob.prob.xl[t]
            Zu[prob.idx_sample[i].x[t]] = prob.prob.xu[t]
        end
    end

    for i = 1:prob.N
        Zl[prob.idx_sample[i].x[T]] = prob.prob.xl[T]
        Zu[prob.idx_sample[i].x[T]] = prob.prob.xu[T]
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
    if prob.prob.stage_constraints > 0
        m_shift = 0
        for t = 1:T-1
            for i = 1:prob.N
                cu[(M_nom + prob.N*(2*prob.prob.n*(prob.prob.T-1) + (T-2)) + prob.N*prob.prob.m*(prob.prob.T-1) + m_shift .+ (1:prob.prob.m_stage[t]))[prob.prob.stage_ineq[t]]] .= Inf
                m_shift += prob.prob.m_stage[t]
            end
        end
    end

    if prob.disturbance_ctrl
        for t = 1:T-1
            for i = 1:prob.N
                cu[(M_nom + prob.M_sample .+ (1:prob.M_dist))] .= Inf
            end
        end
    end
    return cl,cu
end

function eval_objective(prob::SampleProblem,Z)
    (eval_objective(prob.prob,view(Z,prob.idx_nom_z))
        + obj_sample(Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.H,prob.prob.T,
            prob.N,prob.γ)
        + (prob.disturbance_ctrl ? obj_l1(Z,prob.idx_slack,prob.α) : 0.0))
end

function eval_objective_gradient!(∇obj,Z,prob::SampleProblem)
    ∇obj .= 0.0
    eval_objective_gradient!(view(∇obj,prob.idx_nom_z),view(Z,prob.idx_nom_z),
        prob.prob)
    ∇obj_sample!(∇obj,Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.H,prob.prob.T,prob.N,prob.γ)

    prob.disturbance_ctrl && (∇obj_l1!(∇obj,Z,prob.idx_slack,prob.α))
    return nothing
end

function eval_constraint!(c,Z,prob::SampleProblem)
   M_nom = prob.prob.M
   M_sample = prob.M_sample

   eval_constraint!(view(c,1:M_nom),view(Z,prob.idx_nom_z),prob.prob)

   con_sample!(view(c,M_nom .+ (1:M_sample)),Z,prob.idx_nom,prob.idx_sample,prob.idx_x_tmp,
        prob.idx_K,prob.idx_uw,prob.Q,prob.R,prob.models,prob.β,prob.w,
        prob.prob.m_stage,prob.prob.T,prob.N,disturbance_ctrl=prob.disturbance_ctrl)

   prob.disturbance_ctrl && (c_l1!(view(c,M_nom+M_sample .+ (1:prob.M_dist)),Z,prob.idx_uw,prob.idx_slack,prob.prob.T))

   return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::SampleProblem)
    len = length(sparsity_jacobian(prob.prob))
    eval_constraint_jacobian!(view(∇c,1:len),view(Z,prob.idx_nom_z),prob.prob)

    M_nom = prob.prob.M
    M_sample = prob.M_sample

    len_sample = length(sparsity_jacobian_sample(prob.idx_nom,
        prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.idx_uw,prob.prob.m_stage,prob.prob.T,
        prob.N,disturbance_ctrl=prob.disturbance_ctrl))

    ∇con_sample_vec!(view(∇c,len .+ (1:len_sample)),
         Z,prob.idx_nom,
         prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.idx_uw,prob.Q,prob.R,prob.models,
         prob.β,prob.w,prob.prob.m_stage,prob.prob.T,prob.N,disturbance_ctrl=prob.disturbance_ctrl)

    if prob.disturbance_ctrl
        len_dist = length(constraint_l1_sparsity!(prob.idx_uw,prob.idx_slack,prob.prob.T))
        ∇c_l1_vec!(view(∇c,len+len_sample .+ (1:len_dist)),Z,prob.idx_uw,prob.idx_slack,prob.prob.T)
    end
    return nothing
end

function sparsity_jacobian(prob::SampleProblem)
    M_nom = prob.prob.M
    M_sample = prob.M_sample

    if prob.disturbance_ctrl
        collect([sparsity_jacobian(prob.prob)...,
            sparsity_jacobian_sample(prob.idx_nom,
            prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.idx_uw,prob.prob.m_stage,
            prob.prob.T,prob.N,r_shift=M_nom,disturbance_ctrl=prob.disturbance_ctrl)...,
            constraint_l1_sparsity!(prob.idx_uw,prob.idx_slack,prob.prob.T,r_shift=M_nom+M_sample)...])
    else
        collect([sparsity_jacobian(prob.prob)...,
            sparsity_jacobian_sample(prob.idx_nom,
            prob.idx_sample,prob.idx_x_tmp,prob.idx_K,prob.idx_uw,prob.prob.m_stage,
            prob.prob.T,prob.N,r_shift=M_nom,disturbance_ctrl=prob.disturbance_ctrl)...])
    end
end
