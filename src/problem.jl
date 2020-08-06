# z = (x,u,h)
# Z = [z1,...,zT-1,xT]
abstract type Problem end

mutable struct TrajectoryOptimizationProblem <: Problem
    n::Int # states
    m::Int # controls
    T::Int # horizon
    N::Int # number of decision variables
    Nx::Int
    Nu::Int
    Nh::Int
    M::Int # number of constraints
    M_dynamics::Int
    M_stage::Int
    M_general::Int
    x1     # initial state
    xT     # goal state
    ul     # control lower bound
    uu     # control upper bound
    xl     # state lower bound
    xu     # state upper bound
    hl     # time step lower bound
    hu     # time step upper bound
    idx    # indices
    model  # model
    obj    # objective
    goal_constraint
    stage_constraints
    m_stage
    stage_ineq
    general_constraints
    m_general
    general_ineq
end

function init_problem(n,m,T,x1,xT,model,obj;
        ul=[-Inf*ones(m) for t = 1:T-1],
        uu=[Inf*ones(m) for t = 1:T-1],
        xl=[-Inf*ones(n) for t = 1:T],
        xu=[Inf*ones(n) for t = 1:T],
        hl=[-Inf for t = 1:T-1],
        hu=[Inf for t = 1:T-1],
        goal_constraint::Bool=true,
        stage_constraints::Bool=false,
        m_stage=[0 for t=1:T-1],
        stage_ineq=[(1:m_stage[t]) for t=1:T-1],
        general_constraints::Bool=false,
        m_general=0,
        general_ineq=(1:m_general))

    idx = init_indices(n,m,T)

    Nx = n*T
    Nu = m*(T-1)
    Nh = T-1
    N = Nx + Nu + Nh

    M_dynamics = n*(T-1) + (T-2)
    M_stage = stage_constraints*sum(m_stage)
    M_general = general_constraints*m_general
    M = M_dynamics + M_stage + M_general

    return TrajectoryOptimizationProblem(n,m,T,
        N,Nx,Nu,Nh,
        M,M_dynamics,M_stage,M_general,
        x1,xT,
        ul,uu,
        xl,xu,
        hl,hu,
        idx,
        model,
        obj,
        goal_constraint,
        stage_constraints,
        m_stage,
        stage_ineq,
        general_constraints,
        m_general,
        general_ineq)
end

function pack(X0,U0,h0,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    Z0 = zeros(prob.N)
    for t = 1:T-1
        Z0[(t-1)*(n+m+1) .+ (1:n)] = X0[t]
        Z0[(t-1)*(n+m+1)+n .+ (1:m)] = U0[t]
        Z0[(t-1)*(n+m+1)+n+m + 1] = h0
    end
    Z0[(T-1)*(n+m+1) .+ (1:n)] = X0[T]

    return Z0
end

function unpack(Z0,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    X = [Z0[(t-1)*(n+m+1) .+ (1:n)] for t = 1:T]
    U = [Z0[(t-1)*(n+m+1)+n .+ (1:m)] for t = 1:T-1]
    H = [Z0[(t-1)*(n+m+1)+n+m + 1] for t = 1:T-1]

    return X, U, H
end

function init_MOI_Problem(prob::TrajectoryOptimizationProblem)
    return MOIProblem(prob.N,prob.M,prob,
        primal_bounds(prob),constraint_bounds(prob),false)
end


function primal_bounds(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T
    idx = prob.idx

    N = prob.N

    Zl = -Inf*ones(N)
    Zu = Inf*ones(N)

    for t = 1:T-1
        Zl[idx.x[t]] = (t==1 ? prob.x1 : prob.xl[t])
        Zl[idx.u[t]] = prob.ul[t]
        Zl[idx.h[t]] = prob.hl[t]

        Zu[idx.x[t]] = (t==1 ? prob.x1 : prob.xu[t])
        Zu[idx.u[t]] = prob.uu[t]
        Zu[idx.h[t]] = prob.hu[t]
    end

    Zl[idx.x[T]] = (prob.goal_constraint ? prob.xT : prob.xl[T])
    Zu[idx.x[T]] = (prob.goal_constraint ? prob.xT : prob.xu[T])

    return Zl, Zu
end

function constraint_bounds(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T
    idx = prob.idx
    M = prob.M

    cl = zeros(M)
    cu = zeros(M)

    if prob.stage_constraints
        m_shift = 0
        for t = 1:T-1
            cu[(prob.M_dynamics + m_shift .+ (1:prob.m_stage[t]))[prob.stage_ineq[t]]] .= Inf
            m_shift += prob.m_stage[t]
        end
    end

    if prob.general_constraints
        cu[prob.M_dynamics + prob.M_stage .+ (1:prob.M_general)] = prob.general_ineq
    end

    return cl, cu
end

function eval_objective(prob::TrajectoryOptimizationProblem,Z)
    objective(Z,prob)
end

function eval_objective_gradient!(∇l,Z,prob::TrajectoryOptimizationProblem)
    ∇l .= 0.0
    objective_gradient!(∇l,Z,prob)
    return nothing
end

function eval_constraint!(c,Z,prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    dynamics_constraints!(view(c,1:prob.M_dynamics),Z,prob)
    prob.stage_constraints && stage_constraints!(view(c,prob.M_dynamics .+ (1:prob.M_stage)),Z,prob)
    prob.general_constraints && general_constraints!(view(c,prob.M_dyanmics + prob.M_stage .+ (1:prob.m_general)),Z,prob)

    return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    len_dyn_jac = length(sparsity_dynamics_jacobian(prob))
    sparse_dynamics_constraints_jacobian!(view(∇c,1:len_dyn_jac),Z,prob)

    len_stage_jac = length(stage_constraint_sparsity(prob.idx,prob.T,prob.m_stage))
    prob.stage_constraints && ∇stage_constraints!(view(∇c,len_dyn_jac .+ (1:len_stage_jac)),Z,prob)

    len_general_jac = length(general_constraint_sparsity(prob))
    prob.general_constraints && ∇general_constraints!(view(∇c,len_dyn_jac+len_stage_jac .+ (1:len_general_jac)),Z,prob)
    return nothing
end

function sparsity_jacobian(prob::TrajectoryOptimizationProblem)
    n = prob.n
    m = prob.m
    T = prob.T

    sparsity_dynamics = sparsity_dynamics_jacobian(prob)
    sparsity_stage = stage_constraint_sparsity(prob,r_shift=prob.M_dynamics)
    sparsity_general = general_constraint_sparsity(prob,r_shift=prob.M_dynamics+prob.M_stage)
    collect([sparsity_dynamics...,sparsity_stage...,sparsity_general...])
end
