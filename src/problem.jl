# z = (x,u,h)
# Z = [z1,...,zT-1,xT]
abstract type Problem end

mutable struct TrajectoryOptimizationProblem <: Problem
    nx::Int # states
    nu::Int # controls
    T::Int # horizon
    N::Int # number of decision variables
    Nx::Int
    Nu::Int
    Nh::Int
    M::Int # number of constraints
    M_dynamics::Int
    M_contact_dynamics::Int
    M_contact_sdf::Int
    M_contact_med::Int
    M_contact_fc::Int
    M_contact_comp::Int
    M_stage::Int
    M_general::Int
    ul     # control lower bound
    uu     # control upper bound
    xl     # state lower bound
    xu     # state upper bound
    hl     # time step lower bound
    hu     # time step upper bound
    idx    # indices
    model  # model
    obj    # objective
    stage_constraints
    m_stage
    stage_ineq
    general_constraints
    m_general
    general_ineq
    contact_sequence
    T_contact_sequence
end

function init_problem(nx,nu,T,model,obj;
        ul=[-Inf*ones(nu) for t = 1:T-2],
        uu=[Inf*ones(nu) for t = 1:T-2],
        xl=[-Inf*ones(nx) for t = 1:T],
        xu=[Inf*ones(nx) for t = 1:T],
        hl=[-Inf for t = 1:T-2],
        hu=[Inf for t = 1:T-2],
        stage_constraints::Bool=false,
        m_stage=[0 for t=1:T-2],
        stage_ineq=[(1:m_stage[t]) for t=1:T-2],
        general_constraints::Bool=false,
        m_general=0,
        general_ineq=(1:m_general),
        contact_sequence::Bool=false,
        T_contact_sequence=[])

    idx = init_indices(nx,nu,T)

    Nx = nx*T
    Nu = nu*(T-2)
    Nh = T-2
    N = Nx + Nu + Nh

    M_dynamics = nx*(T-2) + (T-3)
    M_contact_sdf = model.nc*T
    M_contact_med = model.nb*(T-2)
    M_contact_fc = model.nc*(T-2)
    M_contact_comp = 3*(T-2)
    M_contact_dynamics = M_contact_sdf + M_contact_med + M_contact_fc + M_contact_comp
    M_stage = stage_constraints*sum(m_stage)
    M_general = general_constraints*m_general
    M = M_dynamics + M_contact_dynamics + M_stage + M_general

    return TrajectoryOptimizationProblem(nx,nu,
        T,
        N,Nx,Nu,Nh,
        M,
        M_dynamics,
        M_contact_dynamics,
        M_contact_sdf,
        M_contact_med,
        M_contact_fc,
        M_contact_comp,
        M_stage,
        M_general,
        ul,uu,
        xl,xu,
        hl,hu,
        idx,
        model,
        obj,
        stage_constraints,
        m_stage,
        stage_ineq,
        general_constraints,
        m_general,
        general_ineq,
        contact_sequence,
        T_contact_sequence)
end

function pack(X0,U0,h0,prob::TrajectoryOptimizationProblem)
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    idx = prob.idx

    Z0 = zeros(prob.N)

    for t = 1:T-2
        Z0[idx.x[t]] = X0[t]
        Z0[idx.u[t]] = U0[t]
        Z0[idx.h[t]] = h0
    end

    Z0[idx.x[T-1]] = X0[T-1]
    Z0[idx.x[T]] = X0[T]

    return Z0
end

function unpack(Z0,prob::TrajectoryOptimizationProblem)
    nx = prob.nx
    nu = prob.nu
    T = prob.T
    idx = prob.idx

    X = [Z0[idx.x[t]] for t = 1:T]
    U = [Z0[idx.u[t]] for t = 1:T-2]
    H = [Z0[idx.h[t]] for t = 1:T-2]

    return X, U, H
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

    for t = 1:T-2
        Zl[idx.x[t]] = prob.xl[t]
        Zl[idx.u[t]] = prob.ul[t]
        Zl[idx.h[t]] = prob.hl[t]

        Zu[idx.x[t]] = prob.xu[t]
        Zu[idx.u[t]] = prob.uu[t]
        Zu[idx.h[t]] = prob.hu[t]
    end

    Zl[idx.x[T-1]] = prob.xl[T-1]
    Zu[idx.x[T-1]] = prob.xu[T-1]

    Zl[idx.x[T]] = prob.xl[T]
    Zu[idx.x[T]] = prob.xu[T]

    # fixed contact sequence
    # if prob.contact_sequence
    #     for t = 1:T-2
    #         if t in prob.T_contact
    #             # Zl[idx.u[t][model.idx_λ]]
    #             # Zu[idx.u[t][model.idx_λ]]
    #             #
    #             # Zl[idx.u[t][model.idx_b]]
    #             # Zu[idx.u[t][model.idx_b]]
    #             #
    #             # Zl[idx.u[t][model.idx_ψ]]
    #             # Zu[idx.u[t][model.idx_ψ]]
    #
    #             Zl[idx.u[t][model.idx_η]] .= 0.0
    #             Zu[idx.u[t][model.idx_η]] .= 0.0
    #         else
    #             Zl[idx.u[t][model.idx_λ]] .= 0.0
    #             Zu[idx.u[t][model.idx_λ]] .= 0.0
    #
    #             Zl[idx.u[t][model.idx_b]] .= 0.0
    #             Zu[idx.u[t][model.idx_b]] .= 0.0
    #
    #             Zl[idx.u[t][model.idx_ψ]] .= 0.0
    #             Zu[idx.u[t][model.idx_ψ]] .= 0.0
    #
    #             # Zl[idx.u[t][model.idx_η]]
    #             # Zu[idx.u[t][model.idx_η]]
    #         end
    #     end
    # end

    return Zl, Zu
end

function constraint_bounds(prob::TrajectoryOptimizationProblem)
    T = prob.T
    M = prob.M

    cl = zeros(M)
    cu = zeros(M)

    # contact dynamics
    # sdf
    cu[prob.M_dynamics .+ (1:prob.M_contact_sdf)] .= Inf
    # med
    cu[prob.M_dynamics+prob.M_contact_sdf .+ (1:prob.M_contact_med)] .= 0.0
    # fc
    cu[prob.M_dynamics+prob.M_contact_sdf+prob.M_contact_med .+ (1:prob.M_contact_fc)] .= Inf
    # comp
    cu[prob.M_dynamics+prob.M_contact_sdf+prob.M_contact_med+prob.M_contact_fc .+ (1:prob.M_contact_comp)] .= Inf

    if prob.stage_constraints
        m_shift = 0
        for t = 1:T-2
            cu[(prob.M_dynamics + prob.M_contact_dynamics + m_shift .+ (1:prob.m_stage[t]))[prob.stage_ineq[t]]] .= Inf
            m_shift += prob.m_stage[t]
        end
    end

    if prob.general_constraints
        cu[(prob.M_dynamics + prob.M_contact_dynamics + prob.M_stage .+ (1:prob.M_general))[prob.general_ineq]] .= Inf
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
    contact_dynamics_constraints!(view(c,prob.M_dynamics .+ (1:prob.M_contact_dynamics)),Z,prob)

    prob.stage_constraints && stage_constraints!(view(c,
        prob.M_dynamics+prob.M_contact_dynamics .+ (1:prob.M_stage)),Z,prob)
    prob.general_constraints && general_constraints!(view(c,
        prob.M_dynamics+prob.M_contact_dynamics + prob.M_stage .+ (1:prob.M_general)),Z,prob)

    return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    len_dyn_jac = length(sparsity_dynamics_jacobian(prob))
    dynamics_constraints_jacobian!(view(∇c,1:len_dyn_jac),Z,prob)

    len_con_dyn_jac = length(sparsity_contact_dynamics_jacobian(prob))
    contact_dynamics_constraints_jacobian!(view(∇c,len_dyn_jac .+ (1:len_con_dyn_jac)),Z,prob)

    len_stage_jac = length(stage_constraint_sparsity(prob))
    prob.stage_constraints && ∇stage_constraints!(view(∇c,len_dyn_jac+len_con_dyn_jac .+ (1:len_stage_jac)),Z,prob)

    len_general_jac = length(general_constraint_sparsity(prob))
    prob.general_constraints && ∇general_constraints!(view(∇c,len_dyn_jac+len_con_dyn_jac+len_stage_jac .+ (1:len_general_jac)),Z,prob)
    return nothing
end

function sparsity_jacobian(prob::TrajectoryOptimizationProblem)
    sparsity_dynamics = sparsity_dynamics_jacobian(prob)
    sparsity_contact_dynamics = sparsity_contact_dynamics_jacobian(prob,
        r_shift=prob.M_dynamics)
    sparsity_stage = stage_constraint_sparsity(prob,
        r_shift=prob.M_dynamics+prob.M_contact_dynamics)
    sparsity_general = general_constraint_sparsity(prob,
        r_shift=prob.M_dynamics+prob.M_contact_dynamics+prob.M_stage)

    collect([sparsity_dynamics...,sparsity_contact_dynamics...,sparsity_stage...,sparsity_general...])
    # collect([sparsity_dynamics...,sparsity_stage...,sparsity_general...])
end
