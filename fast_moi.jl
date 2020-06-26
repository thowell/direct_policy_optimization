using ForwardDiff, FiniteDiff, SparsityDetection, SparseArrays, SparseDiffTools
using BenchmarkTools

include("double_integrator_2D.jl")
include("moi.jl")
include("control.jl")

## Optimize nominal trajectory

# horizon
T = 50

# reference position trajectory
x_ref = range(0.0,stop=1.0,length=T)
y_ref = sin.(range(0.0,stop=2pi,length=T))
z_ref = [[x_ref[t];y_ref[t];0.0;0.0] for t = 1:T]

plot(x_ref,y_ref)
# reference control trajectory
u_ref = [zeros(m) for t = 1:T-1]

# objective
Q = Diagonal(@SVector[10.0,10.0,0.01,0.01])
Qf = Diagonal(@SVector[100.0,100.0,0.01,0.01])
R = 1.0e-2*sparse(I,m,m)

# NLP dimensions
n_nlp = n*T + m*(T-1)
m_nlp = n*T

# NLP problem
prob = Problem(n_nlp,m_nlp,z_ref,u_ref,T,n,m,Q,Qf,R,model,Δt,false)

# NLP initialization
x0 = zeros(n_nlp)
for t = 1:T-1
    x0[(t-1)*(n+m) .+ (1:(n+m))] = [z_ref[t];u_ref[t]]
end
x0[(T-1)*(n+m) .+ (1:(n))] = z_ref[T]


obj(z) = MOI.eval_objective(prob,z)

@benchmark ForwardDiff.gradient($obj,$x0)
@benchmark ForwardDiff.hessian($obj,$x0)
