using Ipopt, MathOptInterface
const MOI = MathOptInterface

struct Problem <: MOI.AbstractNLPEvaluator
    n_nlp
    m_nlp
    obj
    con!
    enable_hessian::Bool
end

function primal_bounds(prob::MOI.AbstractNLPEvaluator)
    x_l = -Inf*ones(prob.n_nlp)
    x_u = Inf*ones(prob.n_nlp)
    return x_l, x_u
end

function constraint_bounds(prob::MOI.AbstractNLPEvaluator)
    c_l = zeros(prob.m_nlp)
    # c_l = -Inf*ones(prob.m_nlp)
    c_u = zeros(prob.m_nlp)
    return c_l, c_u
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    prob.obj(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    grad_f .= ForwardDiff.gradient(prob.obj,x)
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    prob.con!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    jac .= vec(ForwardDiff.jacobian(prob.con!,zeros(prob.m_nlp),x))
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

function sparsity_jacobian(prob::MOI.AbstractNLPEvaluator)

    row = []
    col = []

    r = 1:prob.m_nlp
    c = 1:prob.n_nlp

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function sparsity_hessian(prob::MOI.AbstractNLPEvaluator)

    row = []
    col = []

    r = 1:prob.n_nlp
    c = 1:prob.n_nlp

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob)
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob)
function MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, λ)
    tmp(z) = σ*prob.obj(z) + prob.con!(zeros(eltype(z),prob.m_nlp),z)'*λ
    H .= vec(ForwardDiff.hessian(tmp,x))
    return nothing
end

function solve_ipopt(x0,prob::MOI.AbstractNLPEvaluator)
    x_l, x_u = primal_bounds(prob)
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = 5000
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
