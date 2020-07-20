using LinearAlgebra, ModelingToolkit, ForwardDiff, SparseArrays
using Ipopt
using MathOptInterface
const MOI = MathOptInterface
using BenchmarkTools

T = Float64
n = 100
m = 50

const W = Diagonal(rand(n))
function obj(x)
    transpose(x)*W*x
end

const A = rand(m,n)
const b = rand(m)

function con(x)
    A*x - b
end
function con!(z,x)
    z .= con(x)
end

function L(x,y)
    obj(x) + transpose(y)*c(x)
end

@variables x_sym[1:n], c_sym[1:m]
@parameters y_sym[1:m]

J = obj(x_sym)
obj_fast! = eval(ModelingToolkit.build_function([J],x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇obj_sparsity = ModelingToolkit.sparsejacobian([J],x_sym)
∇obj_fast! = eval(ModelingToolkit.build_function(∇J_sparsity,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇obj_fast = similar(∇J_sparsity,T)

c!(c_sym,x_sym)
c_fast! = eval(ModelingToolkit.build_function(c_sym,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇c_sparsity = ModelingToolkit.sparsejacobian(c_sym,x_sym)
∇c_fast! = eval(ModelingToolkit.build_function(∇c_sparsity,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇c_fast = similar(∇c_sparsity,T)

z = L(x_sym,y_sym)

∇²L_sparsity = ModelingToolkit.sparsehessian(z,x_sym)
∇²L_fast! = eval(ModelingToolkit.build_function(∇²L_sparsity,x_sym,y_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇²L_fast = similar(∇²L_sparsity,T)

x0 = rand(n)
const y0 = rand(m)
const ∇J0 = zeros(n)
const c0 = zeros(m)
const ∇c0 = zeros(m,n)
const ∇²L = zeros(n,n)
#
# @benchmark ForwardDiff.gradient!($∇J0,$f,$x0)
# @benchmark ∇J_fast!($∇J_fast,$x0)
#
# @benchmark c!($c0,$x0)
# @benchmark c_fast!($c0,$x0)
#
# @benchmark ForwardDiff.jacobian!($∇c0,$c!,$c0,$x0)
# @benchmark ∇c_fast!($∇c_fast,$x0)
#
# _L(w) = L(w,y0)
# @benchmark ForwardDiff.hessian!($∇²L,$_L,$x0)
# @benchmark ∇²L_fast!($∇²L_fast,$x0,$y0)
#
# ∇²L_fast!(∇²L_fast,x0,y0)
#
# ∇²L_fast
# ∇²L_fast.nzval

struct Problem <: MOI.AbstractNLPEvaluator
    n
    m
    primal_bounds
    constraint_bounds
    jacobian_sparsity
    hessian_sparsity
    ∇obj_fast
    ∇c_fast
    ∇²L_fast
    enable_hessian::Bool
end

function Problem(n,m,
        primal_bounds,
        constraint_bounds,
        jacobian_sparsity,
        hessian_sparsity,
        ∇obj_fast,∇c_fast,∇²L_fast;
        enable_hessian=true)

    Problem(n,m,
            primal_bounds,
            constraint_bounds,
            jacobian_sparsity,
            hessian_sparsity,
            ∇obj_fast,∇c_fast,∇²L_fast,
            enable_hessian)
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    obj(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    ∇obj_fast!(prob.∇obj_fast,x)
    grad_f .= prob.∇obj_fast.nzval
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    c!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    ∇c_fast!(prob.∇c_fast,x)
    jac .= prob.∇c_fast.nzval
    return nothing
end

function MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, λ)
    ∇²L_fast!(prob.∇²L_fast,x,λ)
    H .= prob.∇²L_fast.nzval
    return nothing
end

function primal_bounds(n)
    x_l = -Inf*ones(n)
    x_u = Inf*ones(n)
    return x_l, x_u
end

function constraint_bounds(m;idx_ineq=(1:0))
    c_l = zeros(m)
    c_l[idx_ineq] .= -Inf
    c_u = zeros(m)
    return c_l, c_u
end

function sparsity(x)
    (row,col,val) = findnz(x)
    collect(zip(row,col))
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.jacobian_sparsity
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = prob.hessian_sparsity

function solve(x0,prob::MOI.AbstractNLPEvaluator;
        tol=1.0e-6,nlp=:ipopt,max_iter=1000)
    x_l, x_u = prob.primal_bounds
    c_l, c_u = prob.constraint_bounds

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    if nlp == :ipopt
        solver = Ipopt.Optimizer()
        solver.options["max_iter"] = max_iter
        solver.options["tol"] = tol
    elseif nlp == :snopt
        solver = SNOPT7.Optimizer()
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
    return
    # Get the solution
    sol = MOI.get(solver, MOI.VariablePrimal(), x)

    return sol
end

prob = Problem(n,m,
    primal_bounds(n),constraint_bounds(m),
    sparsity(∇c_sparsity),sparsity(∇²L_sparsity),
    ∇obj_fast,∇c_fast,∇²L_fast)

MOI.eval_objective(prob, x0)
grad_f = zero(x0)
prob.∇obj_fast
MOI.eval_objective_gradient(prob, grad_f, x0)
g = zeros(m)
MOI.eval_constraint(prob,g,x0)
length(nonzeros(∇c_sparsity))
jac = zeros(nnz(∇c_sparsity))
MOI.eval_constraint_jacobian(prob, jac, x0)
H = zeros(nnz(∇²L_sparsity))
MOI.eval_hessian_lagrangian(prob, H, x0, 0.0, y0)

sol = solve(x0,prob)
