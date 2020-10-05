# z = (x,u)
# Z = [z1,...,zT-1,xT]
abstract type Problem end

mutable struct TrajectoryOptimizationProblem <: Problem
    T::Int # horizon

    N::Int # number of decision variables
    Nx::Int # number of states
    Nu::Int # number of controls

    M::Int # number of constraints
    M_dynamics::Int # number of dynamics constraints
    M_time::Int
    M_stage::Int # number of stage constraints
    M_general::Int # number of general constraints

    ul     # control lower bound
    uu     # control upper bound
    xl     # state lower bound
    xu     # state upper bound

    idx    # indices
    model  # model
    obj    # objective

    free_time
    Δt

    stage_constraints
    m_stage
    stage_ineq
    general_constraints
    m_general
    general_ineq
end

function init_problem(T,model,obj;
        ul=[-Inf*ones(model.nu) for t = 1:T-1],
        uu=[Inf*ones(model.nu) for t = 1:T-1],
        xl=[-Inf*ones(model.nx) for t = 1:T],
        xu=[Inf*ones(model.nx) for t = 1:T],
        free_time::Bool=false,
        Δt=0.0,
        stage_constraints::Bool=false,
        m_stage=[0 for t=1:T-1],
        stage_ineq=[(1:m_stage[t]) for t=1:T-1],
        general_constraints::Bool=false,
        m_general=0,
        general_ineq=(1:m_general))

    idx = init_indices(model.nx,model.nu,T)

    Nx = model.nx*T
    Nu = model.nu*(T-1)
    N = Nx + Nu

    M_dynamics = model.nx*(T-1)
    M_time = free_time*(T-2)
    M_stage = stage_constraints*sum(m_stage)
    M_general = general_constraints*m_general
    M = M_dynamics + M_time + M_stage + M_general

    return TrajectoryOptimizationProblem(
        T,
        N,Nx,Nu,
        M,M_dynamics,M_time,M_stage,M_general,
        ul,uu,
        xl,xu,
        idx,
        model,
        obj,
        free_time,
        Δt,
        stage_constraints,
        m_stage,
        stage_ineq,
        general_constraints,
        m_general,
        general_ineq)
end

function pack(X0,U0,prob::TrajectoryOptimizationProblem)
    nx = prob.model.nx
    nu = prob.model.nu
    T = prob.T
    idx = prob.idx

    Z0 = zeros(prob.N)

    for t = 1:T-1
        Z0[idx.x[t]] = X0[t]
        Z0[idx.u[t]] = U0[t]
    end
    Z0[idx.x[T]] = X0[T]

    return Z0
end

function unpack(Z0,prob::TrajectoryOptimizationProblem)
    nx = prob.model.nx
    nu = prob.model.nu
    T = prob.T
    idx = prob.idx

    X = [Z0[idx.x[t]] for t = 1:T]
    U = [Z0[idx.u[t]] for t = 1:T-1]

    return X, U
end

function init_MOI_Problem(prob::TrajectoryOptimizationProblem)
    return MOIProblem(prob.N,prob.M,prob,
        primal_bounds(prob),constraint_bounds(prob),false)
end


function primal_bounds(prob::TrajectoryOptimizationProblem)

    T = prob.T
    idx = prob.idx

    N = prob.N

    Zl = -Inf*ones(N)
    Zu = Inf*ones(N)

    for t = 1:T-1
        Zl[idx.x[t]] = prob.xl[t]
        Zl[idx.u[t]] = prob.ul[t]

        Zu[idx.x[t]] = prob.xu[t]
        Zu[idx.u[t]] = prob.uu[t]
    end

    Zl[idx.x[T]] = prob.xl[T]
    Zu[idx.x[T]] = prob.xu[T]

    return Zl, Zu
end

function constraint_bounds(prob::TrajectoryOptimizationProblem)
    T = prob.T
    M = prob.M

    cl = zeros(M)
    cu = zeros(M)

    if prob.stage_constraints
        m_shift = 0
        for (t,m_stage) in enumerate(prob.m_stage)
            cu[(prob.M_dynamics + prob.M_time + m_shift .+ (1:m_stage))[prob.stage_ineq[t]]] .= Inf
            m_shift += m_stage
        end
    end

    if prob.general_constraints
        cu[(prob.M_dynamics + prob.M_time + prob.M_stage .+ (1:prob.M_general))[prob.general_ineq]] .= Inf
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
    dynamics_constraints!(view(c,1:prob.M_dynamics),Z,prob)
    prob.free_time && time_constraints!(view(c,
        prob.M_dynamics .+ (1:prob.M_time)),Z,prob)
    prob.stage_constraints && stage_constraints!(view(c,
        prob.M_dynamics + prob.M_time .+ (1:prob.M_stage)),Z,prob)
    prob.general_constraints && general_constraints!(view(c,
        prob.M_dynamics + prob.M_time + prob.M_stage .+ (1:prob.M_general)),Z,prob)

    return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    len_dyn_jac = length(sparsity_dynamics_jacobian(prob))
    sparse_dynamics_constraints_jacobian!(view(∇c,1:len_dyn_jac),Z,prob)

    len_time_jac = length(sparsity_time_jacobian(prob))
    prob.free_time && sparse_time_constraints_jacobian!(view(∇c,
        len_dyn_jac .+ (1:len_time_jac)),Z,prob)

    len_stage_jac = length(stage_constraint_sparsity(prob))
    prob.stage_constraints && ∇stage_constraints!(view(∇c,
        len_dyn_jac + len_time_jac .+ (1:len_stage_jac)),Z,prob)

    len_general_jac = length(general_constraint_sparsity(prob))
    prob.general_constraints && ∇general_constraints!(view(∇c,
        len_dyn_jac + len_time_jac + len_stage_jac .+ (1:len_general_jac)),Z,prob)
    return nothing
end

function sparsity_jacobian(prob::TrajectoryOptimizationProblem)
    sparsity_dynamics = sparsity_dynamics_jacobian(prob)
    sparsity_time = sparsity_time_jacobian(prob,
        r_shift=prob.M_dynamics)
    sparsity_stage = stage_constraint_sparsity(prob,
        r_shift=prob.M_dynamics+prob.M_time)
    sparsity_general = general_constraint_sparsity(prob,
        r_shift=prob.M_dynamics+prob.M_time+prob.M_stage)

    collect([sparsity_dynamics...,sparsity_time...,sparsity_stage...,sparsity_general...])
end
