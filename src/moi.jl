include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

struct MOIProblem <: MOI.AbstractNLPEvaluator
    n
    m
    prob::Problem
    primal_bounds
    constraint_bounds
    enable_hessian::Bool
end

function primal_bounds(prob::MOI.AbstractNLPEvaluator)
    return prob.primal_bounds
end

function constraint_bounds(prob::MOI.AbstractNLPEvaluator)
    return prob.constraint_bounds
end

function sparsity_jacobian(n,m; shift_r=0,shift_c=0)

    row = []
    col = []

    r = shift_r .+ (1:m)
    c = shift_c .+ (1:n)

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    return eval_objective(prob.prob,x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    eval_objective_gradient!(grad_f,x,prob.prob)
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    eval_constraint!(g,x,prob.prob)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    eval_constraint_jacobian!(jac,x,prob.prob)
    return nothing
end

function sparsity_jacobian(prob::MOI.AbstractNLPEvaluator)
    sparsity_jacobian(prob.prob)
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob)
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = []
MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, μ) = nothing

function solve(prob::MOI.AbstractNLPEvaluator,x0;
        tol=1.0e-3,c_tol=1.0e-2,max_iter=1000,nlp=:ipopt,time_limit=120,
        mipl=0,mapl=1)
    x_l, x_u = primal_bounds(prob)
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    if nlp==:ipopt
        solver = Ipopt.Optimizer()
        # solver.options["nlp_scaling_method"] = "none"
        # solver.options["max_cpu_time"] = time_limit
        solver.options["max_iter"] = max_iter
        solver.options["tol"] = tol
        solver.options["constr_viol_tol"] = c_tol
    else
        solver = SNOPT7.Optimizer(Major_feasibility_tolerance=c_tol,
                                  Minor_feasibility_tolerance=tol,
                                  Major_optimality_tolerance=tol,
                                  Time_limit=time_limit,
                                  Major_print_level=mapl,
                                  Minor_print_level=mipl)
    end

    x = MOI.add_variables(solver,prob.n)

    for i = 1:prob.n
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
    return MOI.get(solver, MOI.VariablePrimal(), x)
end
