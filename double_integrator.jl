using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools
include("/home/taylor/Research/non-convex_solver/src/non-convex_solver.jl")

# parameterized model
mutable struct DoubleIntegrator2D
    # z = (x,y,ẋ,ẏ)
    mx
    my
end

# double integrator dynamics (2D)
function dynamics(model::DoubleIntegrator2D,z,u)
    @SVector [z[3], z[4], u[1]/model.mx, u[2]/model.my]
end

n = 4 # number of states
m = 2 # number of controls

model = DoubleIntegrator2D(1.0,1.0)
z0 = zeros(4)
u0 = ones(2)
dynamics(model,z0,u0)

# discrete dynamics (midpoint)
Δt = 0.05
function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end
midpoint(model,z0,u0,Δt)

T = 50
x_ref = 0.5*sin.(range(0.0,stop=2pi,length=T))
y_ref = cos.(range(0.0,stop=pi,length=T))
z_ref = zeros(T*n)
for t = 1:T
    z_ref[(t-1)*n .+ (1:n)] = [x_ref[t];y_ref[t];0.0;0.0]
end
u0 = zeros(m)
u_ref = copy(u0)

# optimize nominal trajectories
z0 = [x_ref[1];y_ref[1];0.0;0.0]
Q = Diagonal(@SVector[1000.0,1000.0,1.0e-4,1.0e-4])
Qf = Diagonal(@SVector[1000.0,1000.0,1.0e-4,1.0e-4])
R = 1.0e-2*sparse(I,m,m)

function f_func(x)
    s = 0.0
    for t = 1:T-1
        z = x[(t-1)*(n+m) .+ (1:n)]
        u = x[(t-1)*(n+m)+n .+ (1:m)]
        s += (z - z_ref[(t-1)*n .+ (1:n)])'*Q*(z - z_ref[(t-1)*n .+ (1:n)]) + (u - u_ref)'*R*(u - u_ref)
    end
    z = x[(T-1)*(n+m) .+ (1:n)]
    s += (z - z_ref[(T-1)*n .+ (1:n)])'*Qf*(z - z_ref[(T-1)*n .+ (1:n)])

    return s
end

f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x)
    c = zeros(eltype(x),n*T)
    for t = 1:T-1
        z = x[(t-1)*(n+m) .+ (1:n)]
        z⁺ = x[t*(n+m) .+ (1:n)]
        u = x[(t-1)*(n+m)+n .+ (1:m)]
        c[(t-1)*n .+ (1:n)] = z⁺ - midpoint(model,z,u,Δt)
    end
    z = x[1:n]
    c[(T-1)*n .+ (1:n)] = z - z0

    return c
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

n_nlp = n*T + m*(T-1)
m_nlp = n*T
xL = -Inf*ones(n_nlp)
xU = Inf*ones(n_nlp)

nlp_model = Model(n_nlp,m_nlp,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)#,cI_idx=zeros(Bool,m_nlp),cA_idx=zeros(Bool,m_nlp))

x0 = zeros(n_nlp)

for t = 1:T-1
    x0[(t-1)*(n+m) .+ (1:(n+m))] = [x_ref[t];y_ref[t];0.0;0.0;u0]
end
x0[(T-1)*(n+m) .+ (1:(n))] = [x_ref[T];y_ref[T];0.0;0.0]

opts = Options{Float64}(kkt_solve=:symmetric,
                        max_iter=250,
                        ϵ_tol=1.0e-5,
                        ϵ_al_tol=1.0e-5,
                        verbose=true,
                        quasi_newton=:none)

s = NonConvexSolver(x0,nlp_model,opts=opts)
@time solve!(s)
x_sol = get_solution(s)

z_nom = [x_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
x_nom = vec([x_sol[(t-1)*(n+m) .+ (1:n)][1] for t = 1:T])
y_nom = vec([x_sol[(t-1)*(n+m) .+ (1:n)][2] for t = 1:T])
u_nom = [x_sol[(t-1)*(n+m)+m .+ (1:m)] for t = 1:T]

plot(x_ref,y_ref,title="reference trajectory",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
scatter!(x_nom,y_nom,color=:orange,label="nominal")

plot(hcat(z_nom...)',width=2.0,label=["x" "y" "ẋ" "ẏ"],title="states")
plot(hcat(u_nom...)',linetype=:steppost,width=2.0,label=["u1" "u2"],title="control")

# discrete TV-LQR
A = []
B = []
for t = 1:T-1
    fz(z) = midpoint(model,z,u_nom[t],Δt)
    fu(u) = midpoint(model,z_nom[t],u,Δt)
    push!(A,ForwardDiff.jacobian(fz,z_nom[t]))
    push!(B,ForwardDiff.jacobian(fu,u_nom[t]))
end
fz(z) = midpoint(model,z,u_nom[T],Δt)
push!(A,ForwardDiff.jacobian(fz,z_nom[T]))

Q_tvlqr = Diagonal(@SVector[100.0,100.0,1.0e-1,1.0e-1])
Qf_tvlqr = Diagonal(@SVector[100.0,100.0,1.0e-1,1.0e-1])
R_tvlqr = 1.0e-5*sparse(I,m,m)
P = []
push!(P,Qf_tvlqr)
K = []

for t = T-1:-1:1
    println(t)
    push!(K,(R_tvlqr + B[t]'*P[end]*B[t])\(B[t]'*P[end]*A[t]))
    push!(P,A[t]'*P[end]*A[t] - (A[t]'*P[end]*B[t])*K[end] + Q_tvlqr)
end


# simulation
model_alt = DoubleIntegrator2D(10.,0.1)

times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)
z_tvlqr_rollout = [z0]
u_tvlqr = []
for tt = 1:T_sim-1
    t = t_sim[tt]
    k = searchsortedlast(times,t)
    w = randn(n)*0.1e-1
    z = z_tvlqr_rollout[end] + w
    u = u_nom[k] - K[k]*(z - z_nom[k])
    push!(z_tvlqr_rollout,midpoint(model_alt,z,u,dt_sim))
    push!(u_tvlqr,u)
end

z_tvlqr_rollout
x_tvlqr_rollout = [z_tvlqr_rollout[t][1] for t = 1:T_sim]
y_tvlqr_rollout = [z_tvlqr_rollout[t][2] for t = 1:T_sim]

plot(x_ref,y_ref,title="simulation",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
scatter!(x_nom,y_nom,color=:orange,label="nominal")
plot!(x_tvlqr_rollout,y_tvlqr_rollout,color=:purple,label="TVLQR",width=2.0)

plot(hcat(u_nom...)',linetype=:steppost,width=2.0,label=["u1 nom." "u2 nom."],title="control")
plot(hcat(u_tvlqr...)',width=2.0,label=["u1 tvlqr" "u2 tvlqr"],title="control")
