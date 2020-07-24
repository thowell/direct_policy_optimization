using LinearAlgebra, ModelingToolkit, ForwardDiff, SparseArrays, StaticArrays
using Ipopt
using MathOptInterface
const MOI = MathOptInterface
using BenchmarkTools

# T = Float64
# n = 100
# m = 50
#
# const W = Diagonal(rand(n))
# function obj(x)
#     transpose(x)*W*x
# end
#
# const A = rand(m,n)
# const b = rand(m)
#
# function con(x)
#     A*x - b
# end
#
# function con(z,x)
#     z .= con(x)
# end

# ∇obj_fast!,∇c_fast!,∇²L_fast!,∇obj_fast,∇c_fast,∇²L_fast,∇c_sparsity,∇²L_sparsity = generate(obj,con!,cy,n,m)

# x0 = rand(n)
# const y0 = rand(m)
# const ∇J0 = zeros(n)
# const c0 = zeros(m)
# const ∇c0 = zeros(m,n)
# const ∇²L = zeros(n,n)
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

function generate(obj,con!,cy,n,m,T)
    @variables x_sym[1:n], c_sym[1:m]
    @parameters y_sym[1:m]

    J = obj(x_sym)
    obj_fast! = eval(ModelingToolkit.build_function([J],x_sym,
                parallel=ModelingToolkit.MultithreadedForm())[2])
    ∇obj_sparsity = ModelingToolkit.sparsejacobian([J],x_sym)
    ∇obj_fast! = eval(ModelingToolkit.build_function(∇obj_sparsity,x_sym,
                parallel=ModelingToolkit.MultithreadedForm())[2])
    ∇obj_fast = similar(∇obj_sparsity,T)

    con!(c_sym,x_sym)
    c_fast! = eval(ModelingToolkit.build_function(c_sym,x_sym,
                parallel=ModelingToolkit.MultithreadedForm())[2])
    ∇c_sparsity = ModelingToolkit.sparsejacobian(c_sym,x_sym)
    ∇c_fast! = eval(ModelingToolkit.build_function(∇c_sparsity,x_sym,
                parallel=ModelingToolkit.MultithreadedForm())[2])
    ∇c_fast = similar(∇c_sparsity,T)

    function L(x,y)
        obj(x) + cy(x,y)
    end

    z = L(x_sym,y_sym)

    ∇²L_sparsity = ModelingToolkit.sparsehessian(z,x_sym)
    ∇²L_fast! = eval(ModelingToolkit.build_function(∇²L_sparsity,x_sym,y_sym,
                parallel=ModelingToolkit.MultithreadedForm())[2])
    ∇²L_fast = similar(∇²L_sparsity,T)

    return ∇obj_fast!,∇c_fast!,∇²L_fast!,∇obj_fast,∇c_fast,∇²L_fast,∇c_sparsity,∇²L_sparsity
end

function obj(x)
    nothing
end

function con!(c,x)
    nothing
end

function cy(x,y)
    nothing
end

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
    con!(g,x)
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

function MOI.features_available(prob::MOI.AbstractNLPEvaluator)
    if prob.enable_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end
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

# prob = Problem(n,m,
#     primal_bounds(n),
#     constraint_bounds(m),
#     sparsity(∇c_sparsity),sparsity(∇²L_sparsity),
#     ∇obj_fast,∇c_fast,∇²L_fast,
#     enable_hessian=true)

# MOI.eval_objective(prob, x0)
# grad_f = zero(x0)
# prob.∇obj_fast
# MOI.eval_objective_gradient(prob, grad_f, x0)
# g = zeros(m)
# MOI.eval_constraint(prob,g,x0)
# length(nonzeros(∇c_sparsity))
# jac = zeros(nnz(∇c_sparsity))
# MOI.eval_constraint_jacobian(prob, jac, x0)
# H = zeros(nnz(∇²L_sparsity))
# MOI.eval_hessian_lagrangian(prob, H, x0, 0.0, y0)

# @time sol = solve(copy(x0),prob)

# g(x,y) = x^2 + y
# f(w) = w[1]^3 + w[2]
# # @register h(x,y)
# @register f(w)
#
# ModelingToolkit.hessian(::typeof(f), w, ::Val) = args[1]^2
#
# F = simplify.(f(z))
#
# f_fast! = eval(ModelingToolkit.build_function([F],z,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# f_sparsity = ModelingToolkit.sparsehessian(F,z)
# ∇²f = eval(ModelingToolkit.build_function(f_sparsity,z,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# _∇²f = similar(f_sparsity,Float64)
#
# x0 = [1.0;1.0]
# ∇²f(_∇²f,x0)
# _∇²f
#
# @variables a
# function huber(a)
#     δ = 1.0
#     if abs(a) <= δ
#         return 0.5*a^2
#     else
#         return δ*(abs(a) - 0.5*δ)
#     end
# end
# da = huber(a)
#
# @register huber(a)
# function ModelingToolkit.derivative(::typeof(huber), args::NTuple{1,Any}, ::Val{1})
#     δ = 1.0
#     a = args[1]
#     if abs(a) <= δ
#         return a
#     else
#         return 1.0
#     end
# end
#
# function ϕ(x)
#     h = 1.0
#     y = _ϕ(x,h)
#     return 1.0*y
# end
#
# function _ϕ(x,h)
#     if x > 0.0
#         return x
#     else
#         return x + h
#     end
# end
#
# @register _ϕ(x,h)
# function ModelingToolkit.derivative(::typeof(_ϕ), args::NTuple{2,Any}, ::Val{1})
#     return 1.0
# end
#
# function ModelingToolkit.derivative(::typeof(_ϕ), args::NTuple{2,Any}, ::Val{2})
#     x = args[1]
#     h = args[2]
#     if x > 0.0
#         return 0.0
#     else
#         return 1.0
#     end
# end
#
# @variables y, w[1:2]
#
# ϕ(y)
# dw = ϕ(y)
# f(x) = transpose(x)*x
#
# dx = f(w)
#
# f_fast! = eval(ModelingToolkit.build_function([dw],w,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# f_sparsity = ModelingToolkit.sparsehessian(dw,w)
# ∇²f = eval(ModelingToolkit.build_function(f_sparsity,w,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# _∇²f = similar(f_sparsity,Float64)
#
#
# n = 2
# function A(x)
#     [x[1] 0.0; 0.0 x[2]]
# end
# using ChainRules
# x0 = rand(2)
#
# A0 = A(x0)
# A0_sqrt = sqrt(A(x0))
# sqrt_vec(A) = vec(sqrt(A))
# sqrt_vec(A0)
# ForwardDiff.jacobian(sqrt_vec,vec(A0))
#
# inv(kron(sqrt(A(x0))',sqrt(A(x0))))
#
# q0 = rand(3)
# @variables q1,q2,q3
# function ϕ(_q1,_q2,_q3)
#     jump_length=1.0
#     # if q[1] < jump_length
#     #     return q[3] - jump_slope*q[1]
#     # else
#     #     return q[3]
#     # end
#     return [_q1]
# end
# @register ϕ(_q1,_q2,_q3)
# dq = ϕ(q1,q2,q3)
#
# f_fast! = eval(ModelingToolkit.build_function(dq,[q1,q2,q3],
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# f_sparsity = ModelingToolkit.sparsejacobian(dq,[q1,q2,q3])
# ∇²f = eval(ModelingToolkit.build_function(f_sparsity,[q1,q2,q3],
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# _∇²f = similar(f_sparsity,Float64)
#
# ∇²f(_∇²f,q0)
# _∇²f
# function ModelingToolkit.derivative(::typeof(ϕ), args::NTuple{3,Any}, ::Val{1})
#     q1_ = args[1]
#     q2_ = args[2]
#     q3_ = args[3]
#     jump_length=1.075
#     # if q1 < jump_length
#     #     return -jump_slope
#     # else
#     #     return 0.0
#     # end
#     return 1.0
#     # return args[1]
#     # return 1.0
# end
#
# function ModelingToolkit.derivative(::typeof(ϕ), args::NTuple{3,Any}, ::Val{2})
#     q1_ = args[1]
#     q2_ = args[2]
#     q3_ = args[3]
#     jump_length=1.0
#     # if q1 < jump_length
#     #     return 0.0
#     # else
#     #     return 0.0
#     # end
#     return 1.0
# end
#
# function ModelingToolkit.derivative(::typeof(ϕ), args::NTuple{3,Any}, ::Val{3})
#     q1_ = args[1]
#     q2_ = args[2]
#     q3_ = args[3]
#     jump_length=1.0
#     # if q1 < jump_length
#     #     return 3.0
#     # else
#     #     return 3.0
#     # end
#     return 3.01
# end
#
# @variables q[1:2]
# q0 = ones(2)
# function ϕ(x)
#     return [x[1];x[2]]
# end
# @register ϕ(x)
# #
# # function _ϕ(x,y)
# #     return [x; y]
# # end
# @register ϕ(x)
# # @register _ϕ(x,y)
#
# (modu, fun, arity) ∈ DiffRules.diffrules()
# function ModelingToolkit.derivative(::typeof(ϕ), args::NTuple{2,Any}, ::Val{1})
#     println("Hello")
#     return 10.0
# end
#
# @variables x y
# myop = sin(x) * y^2
#
# ModelingToolkit.derivative(myop,)
# # function ModelingToolkit.jacobian(::typeof(_ϕ), args::NTuple{2,Any}, ::Val{1})
# #     return 10.0
# # end
#
# dq = ϕ(q)
# # dq_ = _ϕ(q)
#
#
# f_fast! = eval(ModelingToolkit.build_function(dq,q,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# f_sparsity = ModelingToolkit.sparsejacobian(dq,q)
# ∇²f = eval(ModelingToolkit.build_function(f_sparsity,q,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# _∇²f = similar(f_sparsity,Float64)
#
# ∇²f(_∇²f,q0)
# _∇²f

# @variables x
#
# function my_max(y)
#     # return max(x[1],0)
#     return y*y
# end
#
# dx = my_max(x)
#
# my_max(1.0)
#
# f_fast! = eval(ModelingToolkit.build_function([dx],x,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# f_sparsity = ModelingToolkit.sparsejacobian([dx],[x])
# ∇²f = eval(ModelingToolkit.build_function(f_sparsity,x,
#             parallel=ModelingToolkit.MultithreadedForm())[2])
# _∇²f = similar(f_sparsity,Float64)
#
# ∇²f(_∇²f,[2.0])
# _∇²f
#
#
# @register max_max(x)
# function ModelingToolkit.derivative(::typeof(my_max), args::NTuple{1,Any}, ::Val{1})
#     println("hello")
#     return 27.0
# end
#
# @variables A[1:3,1:3]
#
# dA = sqrt(A)



using ModelingToolkit, LinearAlgebra

const n = 2
const N = 2n
@variables x[1:n*N]

function resample(z)
    xμ = zeros(eltype(z),n)
    for i = 1:N
        xμ += z[(i-1)*n .+ (1:n)]
    end
    Σ = sum([(z[(i-1)*n .+ (1:n)] - xμ)*transpose(z[(i-1)*n .+ (1:n)] - xμ) for i = 1:N])
    S = sqrt(Σ)
    return xμ + S[:,1]
end

function Base.sqrt(A::Array{Operation,2})
   A
   eigen(A)
   A
end
x0 = rand(n*N)

resample(x0)
dx = resample(x)
f_fast! = eval(ModelingToolkit.build_function(dx,x,
            parallel=ModelingToolkit.MultithreadedForm())[2])
f_sparsity = ModelingToolkit.sparsejacobian(dx,x)
∇²f = eval(ModelingToolkit.build_function(f_sparsity,x,
            parallel=ModelingToolkit.MultithreadedForm())[2])
_∇²f = similar(f_sparsity,Float64)

∇²f(_∇²f,x0)
_∇²f
