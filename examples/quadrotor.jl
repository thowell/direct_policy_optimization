include("../src/sample_trajectory_optimization.jl")
include("../dynamics/quadrotor.jl")
include("../dynamics/obstacles.jl")

r_quad = 0.1

# Horizon
T = 10

# Bounds
#xl <= x <= xu
xl = -Inf*ones(model.nx)
xl[3] = 0.0

# ul <= u <= uu
uu = 5.0
ul = 0.0

u_hover = 0.5*(-1.0*model.g[3])/4.0*ones(model.nu)

# h = h0 (fixed timestep)
tf0 = 1.0
h0 = tf0/(T-1)
hu = h0
hl = h0

# Initial and final states
x1 = [0.0; 0.0; 10.0; 0.5; 0.5; 0.5; 0.5; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
xT = [1.0; 0.0; 10.0; 0.5; 0.5; 0.5; 0.5; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

# xTu = copy(xT)
# xTu[4:7] .= Inf
# xTl = copy(xTl)
# xTl[4:7] .= -Inf

# Circle obstacle
r = 0.1
xc1 = 0.5
yc1 = 0.05
# xc2 = 0.375
# yc2 = 0.75
# xc3 = 0.25
# yc3 = 0.25
# xc4 = 0.75
# yc4 = 0.75
# xc5 = 0.5
# yc5 = 0.5

# Constraints
function con!(c,x,u)
    c[1] = circle_obs(x[1],x[2],xc1,yc1,r+r_quad)
    # c[2] = circle_obs(x[1],x[2],xc2,yc2,r)
    # c[1] = circle_obs(x[1],x[2],xc3,yc3,r)
    # c[2] = circle_obs(x[1],x[2],xc4,yc4,r)
    # c[1] = circle_obs(x[1],x[2],xc5,yc5,r)
    # c[1:2] = uu*ones(model.nu) - u
    # c[3:4] = u - ul*ones(model.nu)
    nothing
end
m_stage = 1

# Objective
Q = [t < T ? Diagonal([1.0e-1*ones(7);1.0e-3*ones(6)]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[u_hover for t=1:T-1])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0*ones(7);1.0*ones(6)]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    xl=[xl for t=1:T],
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=rk3_implicit,
                    goal_constraint=true,
                    con=con!,
                    m_stage=m_stage
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [u_hover for t = 1:T-1] # random controls

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

    fx(z) = prob.integration(model,x⁺,z,u,h)
    fu(z) = prob.integration(model,x⁺,x,z,h)
    fx⁺(z) = prob.integration(model,z,x,u,h)

    A⁺ = ForwardDiff.jacobian(fx⁺,x⁺)
    push!(A,-A⁺\ForwardDiff.jacobian(fx,x))
    push!(B,-A⁺\ForwardDiff.jacobian(fu,u))
end

K = TVLQR(A,B,Q_lqr,R_lqr)

# Sample
α = 1.0e-4
x11 = x1 + [α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x12 = x1 + [-α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x13 = x1 + [0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x14 = x1 + [0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x15 = x1 + [0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x16 = x1 + [0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

x17 = x1 + [0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x18 = x1 + [0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x19 = x1 + [0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x110 = x1 + [0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x111 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x112 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x113 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x114 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

# normalize quaternion #TODO fix
x17[4:7] = normalize(x17[4:7])
x18[4:7] = normalize(x18[4:7])
x19[4:7] = normalize(x19[4:7])
x110[4:7] = normalize(x110[4:7])
x111[4:7] = normalize(x111[4:7])
x112[4:7] = normalize(x112[4:7])
x113[4:7] = normalize(x113[4:7])
x114[4:7] = normalize(x114[4:7])

x115 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0; 0.0]
x116 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0; 0.0]
x117 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0; 0.0]
x118 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0; 0.0]
x119 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0; 0.0]
x120 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0; 0.0]
x121 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0; 0.0]
x122 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0; 0.0]
x123 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α; 0.0]
x124 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α; 0.0]
x125 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; α]
x126 = x1 + [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; -α]

x1_sample = [x11,x12,x13,x14,x15,x16,x17,x18,x19,x110,x111,x112,x113,x114,
    x115,x116,x117,x118,x119,x120,x121,x122,x123,x124,x125,x126]

N = length(x1_sample)
models = [model for i = 1:N]
K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 1.0e-4*ones(model.nx)
γ = 1.0

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,β=β,w=w,γ=γ)
prob_sample_moi = init_MOI_Problem(prob_sample)

# Z0_sample = pack(X0,U0,h0,K0,prob_sample)
Z0_sample = pack(X_nom,U_nom,H_nom[1],K,prob_sample)

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
using Plots
# Position trajectory
x_nom_pos = [X_nom[t][1] for t = 1:T]
y_nom_pos = [X_nom[t][2] for t = 1:T]
z_nom_pos = [X_nom[t][3] for t = 1:T]
q_nom = [X_nom[t][4:7] for t = 1:T]
q_nom_norm = [norm(q_nom[t]) for t =1:T]
plot(z_nom_pos)
plot(q_nom_norm)
pts = Plots.partialcircle(0,2π,100,r)
cx,cy = Plots.unzip(pts)
cx1 = [_cx + xc1 for _cx in cx]
cy1 = [_cy + yc1 for _cy in cy]
# cx2 = [_cx + xc2 for _cx in cx]
# cy2 = [_cy + yc2 for _cy in cy]
# cx3 = [_cx + xc3 for _cx in cx]
# cy3 = [_cy + yc3 for _cy in cy]
# cx4 = [_cx + xc4 for _cx in cx]
# cy4 = [_cy + yc4 for _cy in cy]
# cx5 = [_cx + xc5 for _cx in cx]
# cy5 = [_cy + yc5 for _cy in cy]

plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
# plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
# plt = plot(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
# plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
# plt = plot(Shape(cx5,cy5),color=:red,label="",linecolor=:red)
plt = plot!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=2.0,label="nominal (tf=$(round(sum(H_nom),digits=3))s)",color=:purple,legend=:topleft)
x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
y_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,width=2.0,label="sample  (tf=$(round(sum(H_nom_sample),digits=3))s)",color=:orange,legend=:bottomright)
savefig(plt,joinpath(@__DIR__,"results/quadrotor_xy_trajectory_T10.png"))

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',color=:purple,width=2.0,
    title="Quadrotor",xlabel="time (s)",ylabel="control",label="",
    legend=:bottom,linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,label="",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/quadrotor_control.png"))

# Samples

# State samples
plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
for i = 1:N
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1:3,:]',label="");
end
plt1 = plot!(t_sample,hcat(X_nom_sample...)[1:3,:]',color=:red,width=2.0,
    label=["nominal" "" ""])
display(plt1)
savefig(plt1,joinpath(@__DIR__,"results/quadrotor_sample_states.png"))

# Control samples
plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
for i = 1:N
    plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
        linetype=:steppost);
end
plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
    label=["nominal" ""],linetype=:steppost)
display(plt2)
savefig(plt2,joinpath(@__DIR__,"results/quadrotor_sample_control.png"))
