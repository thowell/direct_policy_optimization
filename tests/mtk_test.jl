using LinearAlgebra, ForwardDiff, SparseArrays, StaticArrays
using ModelingToolkit
using BenchmarkTools

println("number of threads: $(Threads.nthreads())")

# set up QP
T = Float64

# problem size
n = 100
m = 50

# problem data
const W = Diagonal(@SVector rand(n))
const A = rand(m,n)
const b = rand(m)

# methods
function obj(x)
    transpose(x)*W*x
end

function con(x)
    A*x - b
end

function con!(z,x)
    z .= con(x)
    nothing
end

function L(x,y)
    obj(x) + transpose(y)*con(x)
end

# values
const x0 = rand(n)
const y0 = rand(m)
const c0 = zeros(m)

# test methods
@benchmark obj($x0)
@benchmark con($x0)
@benchmark con!($c0,$x0)
@benchmark L($x0,$y0)

# symbolic variables
@variables x_sym[1:n], c_sym[1:m]
@parameters y_sym[1:m]

# generate fast methods and sparsity, and allocate memory
J = obj(x_sym);
obj_fast! = eval(ModelingToolkit.build_function([J],x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇obj_sparsity = ModelingToolkit.sparsejacobian([J],x_sym)
∇obj_fast! = eval(ModelingToolkit.build_function(∇obj_sparsity,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇obj_fast = similar(∇obj_sparsity,T)

con!(c_sym,x_sym);
c_fast! = eval(ModelingToolkit.build_function(c_sym,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇c_sparsity = ModelingToolkit.sparsejacobian(c_sym,x_sym)
∇c_fast! = eval(ModelingToolkit.build_function(∇c_sparsity,x_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇c_fast = similar(∇c_sparsity,T)

z = L(x_sym,y_sym);
∇²L_sparsity = ModelingToolkit.sparsehessian(z,x_sym)
∇²L_fast! = eval(ModelingToolkit.build_function(∇²L_sparsity,x_sym,y_sym,
            parallel=ModelingToolkit.MultithreadedForm())[2])
∇²L_fast = similar(∇²L_sparsity,T)

# compare performance
@benchmark ForwardDiff.gradient!($∇obj_fast,$obj,$x0)
@benchmark ∇obj_fast!($∇obj_fast,$x0)

@benchmark con!($c0,$x0)
@benchmark c_fast!($c0,$x0)

@benchmark ForwardDiff.jacobian!($∇c_fast,$con!,$c0,$x0)
@benchmark ∇c_fast!($∇c_fast,$x0)

_L(w) = L(w,y0)
@benchmark ForwardDiff.hessian!($∇²L_fast,$_L,$x0)
@benchmark ∇²L_fast!($∇²L_fast,$x0,$y0)

# ∇²L_fast!(∇²L_fast,x0,y0)
# ∇²L_fast
# ∇²L_fast.nzval
