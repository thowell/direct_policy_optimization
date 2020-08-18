include("../src/sample_trajectory_optimization.jl")
include("../dynamics/acrobot.jl")
using Plots

# Horizon
T = 51

# Bounds

# ul <= u <= uu
uu = 10.0
ul = -10.0

# hl <= h <= hu
tf0 = 5.0
h0 = tf0/(T-1) # timestep

hu = h0
hl = h0

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]

# Objective
Q = [t<T ? Diagonal(ones(model.nx)) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-3*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal(ones(model.nx)) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Initialization
X0 = linear_interp(x1,xT,T) # linear interpolation for states
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

plot(hcat(X_nominal...)',xlabel="time step")
plot(hcat(U_nominal...)',xlabel="time step")


# using Colors
# using CoordinateTransformations
# using FileIO
# using GeometryTypes:Vec,HyperRectangle,HyperSphere,Point3f0,Cylinder
# using LinearAlgebra
# using MeshCat, MeshCatMechanisms
# using MeshIO
# using Rotations
# using RigidBodyDynamics
#
# function cable_transform(y,z)
#     v1 = [0,0,1]
#     v2 = y[1:3,1] - z[1:3,1]
#     normalize!(v2)
#     ax = cross(v1,v2)
#     ang = acos(v1'v2)
#     R = AngleAxis(ang,ax...)
#
#     if any(isnan.(R))
#         R = I
#     else
#         nothing
#     end
#
#     compose(Translation(z),LinearMap(R))
# end
#
# vis = Visualizer()
# open(vis)
visualize!(vis,model,X_nominal,Δt=h0)

# Samples
N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-3*ones(model.nx)
γ = 1.0

α = 1.0e-3
x11 = α*[1.0; 0.0]
x12 = α*[-1.0; 0.0]
x13 = α*[0.0; 1.0]
x14 = α*[0.0; -1.0]
x1_sample = resample([x11,x12,x13,x14],β=β,w=w)
K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ)
prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,nlp=:SNOPT7)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

# Plot results

# Time
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nominal))s")
display("time (sample nominal): $(sum(H_nom_sample))s")

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nominal...))',
    color=:purple,width=2.0,title="Pendulum",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/pendulum_stagetrol_noise.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ (nominal)",title="Pendulum",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="x (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="x (sample)")
savefig(plt,joinpath(@__DIR__,"results/pendulum_state_noise.png"))

# State samples
plt1 = plot(t_sample,hcat(X_nom_sample...)[1,:],color=:red,width=2.0,title="",
    label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1,:],label="");
end

plt2 = plot(t_sample,hcat(X_nom_sample...)[2,:],color=:red,width=2.0,label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt2 = plot!(t_sample,hcat(X_sample[i]...)[2,:],label="");
end
plt12 = plot(plt1,plt2,layout=(2,1),title=["θ" "x"],xlabel="time (s)")
savefig(plt,joinpath(@__DIR__,"results/pendulum_sample_state.png"))

# Control samples
plt3 = plot(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=2.0,
    title="Control",label="",xlabel="time (s)");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1,:],label="");
end
display(plt3)
savefig(plt,joinpath(@__DIR__,"results/pendulum_sample_control.png"))
