using LinearAlgebra, Plots, ForwardDiff, Ipopt, MathOptInterface
const MOI = MathOptInterface

struct Problem <: MOI.AbstractNLPEvaluator
    n_nlp
    m_nlp
    z_nom
    u_nom
    T
    n
    m
    Q
    Qf
    R
    model
    Δt
    enable_hessian::Bool
end

function primal_bounds(prob::MOI.AbstractNLPEvaluator)
    x_l = -Inf*ones(prob.n_nlp)
    x_u = Inf*ones(prob.n_nlp)
    return x_l, x_u
end

function constraint_bounds(prob::MOI.AbstractNLPEvaluator)
    c_l = zeros(prob.m_nlp)
    c_u = zeros(prob.m_nlp)
    return c_l, c_u
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    s = 0.0
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    R = prob.R
    z_nom = prob.z_nom
    u_nom = prob.u_nom

    for t = 1:T
        z = x[(t-1)*(n + m) .+ (1:n)]
        if t < T
            u = x[(t-1)*(n + m)+n .+ (1:m)]
            s += (z - z_nom[t])'*Q*(z - z_nom[t]) + (u - u_nom[t])'*R*(u - u_nom[t])
        else
            s += (z - z_nom[t])'*Qf*(z - z_nom[t])
        end
    end
    return s
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom

    for t = 1:T
        z = x[(t-1)*(n + m) .+ (1:n)]
        if t < T
            grad_f[(t-1)*(n + m) .+ (1:n)] = 2.0*Q*(z - z_nom[t])
            u = x[(t-1)*(n + m)+n .+ (1:m)]

            grad_f[(t-1)*(n + m) + n .+ (1:m)] = 2.0*R*(u - u_nom[t])
        else
            grad_f[(t-1)*(n + m) .+ (1:n)] = 2.0*Qf*(z - z_nom[t])
        end
    end
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt

    for t = 1:T-1
        z = x[(t-1)*(n + m) .+ (1:n)]
        z⁺ = x[t*(n + m) .+ (1:n)]
        u = x[(t-1)*(n + m) + n .+ (1:m)]
        g[(t-1)*n .+ (1:n)] = z⁺ - midpoint(model,z,u,prob.Δt)
    end

    z = x[1:n]
    g[(T-1)*n .+ (1:n)] = z - z_nom[1]
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt

    JAC = zeros(prob.m_nlp,prob.n_nlp)
    for t = 1:T-1
        r_idx = (t-1)*n .+ (1:n)
        c1_idx = (t-1)*(n + m) .+ (1:n)
        c2_idx = t*(n + m) .+ (1:n)
        c3_idx = (t-1)*(n + m) + n .+ (1:m)

        z = x[(t-1)*(n + m) .+ (1:n)]
        z⁺ = x[t*(n + m) .+ (1:n)]
        u = x[(t-1)*(n + m)+n .+ (1:m)]

        f1(w) = z⁺ - midpoint(model,w,u,prob.Δt)
        # f2(w) = z⁺ - midpoint(model,z,u,prob.Δt)
        f3(w) = z⁺ - midpoint(model,z,w,prob.Δt)

        JAC[r_idx,c1_idx] = ForwardDiff.jacobian(f1,z)
        JAC[CartesianIndex.(r_idx,c2_idx)] .= 1.0
        JAC[r_idx,c3_idx] = ForwardDiff.jacobian(f3,u)
    end
    r_idx = (T-1)*n .+ (1:n)
    c_idx = 1:n
    JAC[CartesianIndex.(r_idx,c_idx)] .= 1.0

    jac .= vec(JAC)
    return nothing
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function sparsity(prob::MOI.AbstractNLPEvaluator)

    row = []
    col = []

    r = 1:prob.m_nlp
    c = 1:prob.n_nlp

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity(prob)
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = []
MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, μ) = nothing


function solve_ipopt(x0,prob::MOI.AbstractNLPEvaluator)
    x_l, x_u = primal_bounds(prob)
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = 1000
    # solver.options["tol"] = 1.0e-3

    x = MOI.add_variables(solver,prob.n_nlp)

    for i = 1:prob.n_nlp
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res
end
