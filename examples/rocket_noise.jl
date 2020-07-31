include("../src/sample_trajectory_optimization.jl")
include("../dynamics/rocket.jl")
using Plots

# Horizon
T = 30

# Bounds

# ul <= u <= uu
uu = [6500.0;130;15.0]
ul = [0.0;-130.0;15.0]

# h = h0 (fixed timestep)
tf0 = 30.0
h0 = tf0/(T-1)
hu = 5*h0
hl = 0.0

# Initial and final states
x1 = [100.0; 100.0; 5.0; 10.0; 10.0; 0.1]
xT = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

# Objective
Q = [t < T ? Diagonal(zeros(model.nx)) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal(zeros(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=rk3_implicit,
                    goal_constraint=true
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

# Sample
α = 1.0e-4
x11 = α*[1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x12 = α*[-1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x13 = α*[0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
x14 = α*[0.0; -1.0; 0.0; 0.0; 0.0; 0.0]
x15 = α*[0.0; 0.0; 1.0; 0.0; 0.0; 0.0]
x16 = α*[0.0; 0.0; -1.0; 0.0; 0.0; 0.0]
x17 = α*[0.0; 0.0; 0.0; 1.0; 0.0; 0.0]
x18 = α*[0.0; 0.0; 0.0; -1.0; 0.0; 0.0]
x19 = α*[0.0; 0.0; 0.0; 0.0; 1.0; 0.0]
x110 = α*[0.0; 0.0; 0.0; 0.0; -1.0; 0.0]
x111 = α*[0.0; 0.0; 0.0; 0.0; 0.0; 1.0]
x112 = α*[0.0; 0.0; 0.0; 0.0; 0.0; -1.0]
x1_sample = [x11,x12,x13,x14,x15,x16]

N = length(x1_sample)
models = [model for i = 1:N]
K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 2.5e-4*ones(model.nx)
γ = 1.0

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,β=β,w=w,γ=γ)
prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X0,U0,h0,K0,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample))

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
y_nom_pos = [X_nom[t][2] for t = 1:T]
pts = Plots.partialcircle(0,2π,100,r)
cx,cy = Plots.unzip(pts)
# cx1 = [_cx + xc1 for _cx in cx]
# cy1 = [_cy + yc1 for _cy in cy]
# cx2 = [_cx + xc2 for _cx in cx]
# cy2 = [_cy + yc2 for _cy in cy]
cx3 = [_cx + xc3 for _cx in cx]
cy3 = [_cy + yc3 for _cy in cy]
cx4 = [_cx + xc4 for _cx in cx]
cy4 = [_cy + yc4 for _cy in cy]
cx5 = [_cx + xc5 for _cx in cx]
cy5 = [_cy + yc5 for _cy in cy]

# plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
# plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
# plt = plot(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
# plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
plt = plot(Shape(cx5,cy5),color=:red,label="",linecolor=:red)
plt = plot!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=2.0,label="nominal (tf=$(round(sum(H_nom),digits=3))s)",color=:purple,legend=:topleft)
x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
y_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,width=2.0,label="sample  (tf=$(round(sum(H_nom_sample),digits=3))s)",color=:orange,legend=:bottomright)
savefig(plt,joinpath(@__DIR__,"results/dubins_trajectory.png"))

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',color=:purple,width=2.0,
    title="Dubins",xlabel="time (s)",ylabel="control",label=["v (nominal)" "ω (nominal)"],
    legend=:bottom,linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,label=["v (sample)" "ω (sample)"],linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/dubins_control.png"))

# Samples

# State samples
plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
for i = 1:N
    plt1 = plot!(t_sample,hcat(X_sample[i]...)',label="");
end
plt1 = plot!(t_sample,hcat(X_nom_sample...)',color=:red,width=2.0,
    label=["nominal" "" ""])
display(plt1)
savefig(plt1,joinpath(@__DIR__,"results/dubins_sample_states.png"))

# Control samples
plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
for i = 1:N
    plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
        linetype=:steppost);
end
plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
    label=["nominal" ""],linetype=:steppost)
display(plt2)
savefig(plt2,joinpath(@__DIR__,"results/dubins_sample_controls.png"))
