include("../src/sample_motion_planning.jl")
include("../dynamics/rocket.jl")
using Plots

# Horizon
T = 10

# Initial and final states
x1 = [0.5; model.l2 + 1.0; 0.0; 0.0; 0.0; 0.1; 0.0; 0.0]
xT = [0.0; model.l2; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

# Bounds

# xl <= x <= xl
xl = -Inf*ones(model.nx)
xl[2] = model.l2
xl[3] = -30.0*pi/180.0
xl[4] = -20.0*pi/180.0

xu = Inf*ones(model.nx)
xu[2] = x1[2]
xu[3] = 30.0*pi/180.0
xu[4] = 20.0*pi/180.0

# ul <= u <= uu
uu = [100.0;50.0;15.0*pi/180.0]
ul = [0.0;-50.0;-15.0*pi/180.0]

# h = h0 (fixed timestep)
tf0 = 1.0
h0 = tf0/(T-1)
hu = h0
hl = h0

# Objective
Q = [t < T ? Diagonal([1.0*ones(4);1.0*ones(4)]) : Diagonal(ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-2*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0*ones(4);10.0*ones(4)]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    xl=[xl for t=1:T],
                    xu=[xu for t=1:T],
                    ul=[ul for t=1:T-1],
                    uu=[uu for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [1.0*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

# TVLQR policy
A = []
B = []
for t = 1:T-1
    x = X_nom[t]
    u = U_nom[t]
    h = H_nom[t]
    x⁺ = X_nom[t+1]

    fx(z) = discrete_dynamics(model,x⁺,z,u,h)
    fu(z) = discrete_dynamics(model,x⁺,x,z,h)
    fx⁺(z) = discrete_dynamics(model,z,x,u,h)

    A⁺ = ForwardDiff.jacobian(fx⁺,x⁺)
    push!(A,-A⁺\ForwardDiff.jacobian(fx,x))
    push!(B,-A⁺\ForwardDiff.jacobian(fu,u))
end

K = TVLQR(A,B,Q_lqr,R_lqr)

# Sample
α = 5.0e-4
x11 = x1 + α*[1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x12 = x1 + α*[-1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x13 = x1 + α*[0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x14 = x1 + α*[0.0; -1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x15 = x1 + α*[0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x16 = x1 + α*[0.0; 0.0; -1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x17 = x1 + α*[0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
x18 = x1 + α*[0.0; 0.0; 0.0; -1.0; 0.0; 0.0; 0.0; 0.0]
x19 = x1 + α*[0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0]
x110 = x1 + α*[0.0; 0.0; 0.0; 0.0; -1.0; 0.0; 0.0; 0.0]
x111 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0]
x112 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; -1.0; 0.0; 0.0]
x113 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0]
x114 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -1.0; 0.0]
x115 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0]
x116 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -1.0]
x1_sample = [x11,x12,x13,x14,x15,x16,x17,x18,x19,x110,x111,x112,x113,x114,x115,x116]

model1 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model2 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model3 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model4 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model5 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model6 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model7 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model8 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model9 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model10 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model11 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model12 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model13 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model14 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model15 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)
model16 = Rocket(1.0,0.1,0.001,9.81,1.0,1.0,0.01,nx,nu)


N = length(x1_sample)
models = [model for i = 1:N]
K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 5.0e-4*ones(model.nx)
γ = 1.0

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,β=β,w=w,γ=γ)
prob_sample_moi = init_MOI_Problem(prob_sample)

# Z0_sample = pack(X0,U0,h0,K0,prob_sample)
Z0_sample = pack(X_nom,U_nom,h0,K,prob_sample)
Z_sample_sol = Z0_sample

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),max_iter=500)

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)

# Time trajectories
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nom))s")
display("time (sample): $(sum(H_nom_sample))s")

# Plots results

# Position trajectory
x_nom_pos = [X_nom[t][1] for t = 1:T]
z_nom_pos = [X_nom[t][2] for t = 1:T]
θ_nom_pos = [X_nom[t][3] for t = 1:T]
plot(θ_nom_pos)

plt = plot(x_nom_pos,z_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,label="nominal (tf=$(round(sum(H_nom),digits=3))s)",color=:purple,legend=:topleft)

x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
z_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,z_sample_pos,aspect_ratio=:equal,width=2.0,label="sample  (tf=$(round(sum(H_nom_sample),digits=3))s)",color=:orange,legend=:bottomright)
savefig(plt,joinpath(@__DIR__,"results/rocket_trajectory.png"))

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',width=2.0,
    title="Rocket",xlabel="time (s)",ylabel="control",
    label=["FE (nominal)" "FT (nominal)" "φ (nominal)"],color=:purple,
    legend=:top,linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,label=["FE (sample nominal)" "FT (sample nominal)" "φ (sample nominal)"],linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/rocket_control.png"))
#
# # Samples

# State samples
plt1 = plot(title="Sample states",legend=:topright,xlabel="time (s)");
for i = 1:N
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1:3,:]',label="");
end
plt1 = plot!(t_sample,hcat(X_nom_sample...)[1:3,:]',color=:red,width=2.0,
    label=label=["x" "z" "θ"])
display(plt1)
savefig(plt1,joinpath(@__DIR__,"results/rocket_sample_states.png"))

# Control samples
plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:topleft);
for i = 1:N
    plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
        linetype=:steppost);
end
plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
    label=["FE (sample nominal)" "FT (sample nominal)" "φ (sample nominal)"],linetype=:steppost)
display(plt2)
savefig(plt2,joinpath(@__DIR__,"results/rocket_sample_controls.png"))

using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshIO
using Rotations

vis = Visualizer()
open(vis)

l1 = Cylinder(Point3f0(0,0,-model.l2),Point3f0(0,0,model.l1),convert(Float32,0.1))
setobject!(vis["rocket"],l1,MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

anim = MeshCat.Animation(convert(Int,floor(1/h0)))

for t = 1:T
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(X_nom[t][1],0.0,X_nom[t][2]),LinearMap(RotY(X_nom[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)
# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
