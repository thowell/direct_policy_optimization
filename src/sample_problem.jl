mutable struct SampleProblem <: Problem
    prob::TrajectoryOptimizationProblem

    u_ctrl

    N_nlp::Int # number of decision variables
    Nx::Int
    Nu::Int
    Nh::Int
    Nxs::Int
    NK::Int
    Nuw::Int

    M_nlp::Int # number of constraints
    M_dynamics::Int # number of sample constraints
    M_ctrl::Int
    M_stage::Int
    M_uw::Int

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
end

function init_sample_problem(prob::TrajectoryOptimizationProblem,models,x1,Q,R,H;
        u_ctrl=(1:prob.m),
        β=1.0,w=1.0,γ=1.0,
        disturbance_ctrl=false,α=1.0)

    nx = prob.n
    nu = prob.m
    nu_ctrl = length(u_ctrl)

    T = prob.T
    N = length(models)
    @assert N == 2*nx
    @assert size(R[1],1) == nu

    Nx = N*(nx*T)
    Nu = N*nu*(T-1)
    Nh = N*(T-1)
    Nxs = N*(nx*(T-1))
    NK = nu*nx*(T-1)
    Nuw = disturbance_ctrl*2*N*nx*(T-1)
    N_nlp = prob.N + Nx + Nu + Nh + Nxs + NK + Nuw

    M_dynamics = N*(2*nx*(T-1) + (T-2))
    M_ctrl = N*nu_ctrl*(T-1)
    M_stage = prob.stage_constraints*N*sum(prob.m_stage)
    M_uw = disturbance_ctrl*2*N*nx*(T-1)

    M_nlp = prob.M + M_dynamics + M_ctrl + M_stage + M_uw

    idx_nom = init_indices(nx,nu,T,time=true,shift=0)
    idx_nom_z = 1:prob.N
    shift = prob.N
    idx_sample = [init_indices(nx,nu,T,time=true,shift=shift + (i-1)*(prob.N)) for i = 1:N]
    shift += N*prob.N
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
        idx_uw = [[(1:0) for t = 1:T-1] for i = 1:N]
        idx_slack = [[(1:0) for t = 1:T-1] for i = 1:N]
    end

    return SampleProblem(
        prob,
        u_ctrl,
        N_nlp,Nx,Nu,Nh,Nxs,NK,Nuw,
        M_nlp,M_dynamics,M_ctrl,M_stage,M_uw,
        idx_nom,idx_nom_z,
        idx_sample,idx_x_tmp,idx_K,
        Q,R,H,
        N,models,x1,β,w,γ,
        disturbance_ctrl,
        α,
        idx_uw,
        idx_slack
        )
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

function controller(Z0,prob::SampleProblem)
    T = prob.prob.T
    N = prob.N

    X_nom = [Z0[prob.idx_nom.x[t]] for t = 1:T]
    U_nom = [Z0[prob.idx_nom.u[t]] for t = 1:T-1]
    H_nom = [Z0[prob.idx_nom.h[t]] for t = 1:T-1]

    K_sample = [reshape(Z0[prob.idx_K[t]],prob.prob.m,prob.prob.n) for t = 1:T-1]

    return K_sample, X_nom, U_nom, H_nom
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
                cu[(M_nom + prob.M_dynamics + prob.M_ctrl + m_shift .+ (1:prob.prob.m_stage[t]))[prob.prob.stage_ineq[t]]] .= Inf
                m_shift += prob.prob.m_stage[t]
            end
        end
    end

    if prob.disturbance_ctrl
        for t = 1:T-1
            for i = 1:prob.N
                cu[(M_nom + prob.M_dynamics + prob.M_ctrl + prob.M_stage .+ (1:prob.M_uw))] .= Inf
            end
        end
    end
    return cl,cu
end

function eval_objective(prob::SampleProblem,Z)
    (eval_objective(prob.prob,view(Z,prob.idx_nom_z))
        + obj_sample(Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.H,prob.prob.T,
            prob.N,prob.γ)
        + (prob.disturbance_ctrl ? obj_l1(Z,prob) : 0.0))
end

function eval_objective_gradient!(∇obj,Z,prob::SampleProblem)
    ∇obj .= 0.0
    eval_objective_gradient!(view(∇obj,prob.idx_nom_z),view(Z,prob.idx_nom_z),
        prob.prob)
    ∇obj_sample!(∇obj,Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.H,prob.prob.T,prob.N,prob.γ)

    prob.disturbance_ctrl && (∇obj_l1!(∇obj,Z,prob))
    return nothing
end

function eval_constraint!(c,Z,prob::SampleProblem)
   M = prob.prob.M

   eval_constraint!(view(c,1:M),view(Z,prob.idx_nom_z),prob.prob)

   sample_dynamics_constraints!(view(c,M .+ (1:prob.M_dynamics)),Z,prob)
   sample_control_constraints!(view(c,M+prob.M_dynamics .+ (1:prob.M_ctrl)),Z,prob)
   prob.prob.stage_constraints && sample_stage_constraints!(view(c,M+prob.M_dynamics+prob.M_ctrl .+ (1:prob.M_stage)),Z,prob)

   prob.disturbance_ctrl && (sample_disturbance_constraints!(view(c,M+prob.M_dynamics+prob.M_ctrl+prob.M_stage .+ (1:prob.M_uw)),Z,prob))

   return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::SampleProblem)
    len = length(sparsity_jacobian(prob.prob))
    eval_constraint_jacobian!(view(∇c,1:len),view(Z,prob.idx_nom_z),prob.prob)

    M = prob.prob.M

    len_dyn = length(sparsity_jacobian_sample_dynamics(prob))
    ∇sample_dynamics_constraints!(view(∇c,len .+ (1:len_dyn)),Z,prob)

    len_ctrl = length(sparsity_jacobian_sample_control(prob))
    ∇sample_control_constraints!(view(∇c,len+len_dyn .+ (1:len_ctrl)),Z,prob)

    len_stage = length(sparsity_jacobian_sample_stage(prob))
    prob.prob.stage_constraints && ∇sample_stage_constraints!(view(∇c,len+len_dyn+len_ctrl .+ (1:len_stage)),Z,prob)

    if prob.disturbance_ctrl
        len_dist = length(sparsity_jacobian_sample_disturbance!(prob))
        ∇sample_disturbance_constraints!(view(∇c,len+len_dyn+len_ctrl+len_stage .+ (1:len_dist)),Z,prob)
    end
    return nothing
end

function sparsity_jacobian(prob::SampleProblem)
    collect([sparsity_jacobian(prob.prob)...,
             sparsity_jacobian_sample_dynamics(prob,r_shift=prob.prob.M)...,
             sparsity_jacobian_sample_control(prob,
                r_shift=prob.prob.M+prob.M_dynamics)...,
             sparsity_jacobian_sample_stage(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_ctrl)...,
             sparsity_jacobian_sample_disturbance(prob,
                r_shift=prob.prob.M+prob.M_dynamics+prob.M_ctrl+prob.M_stage)...])
end
