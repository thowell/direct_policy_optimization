include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/rocket.jl"))
using Plots

# Model
model = model
nx = model.nx
nu = model.nu

# Horizon
T = 51

# Initial and final states
x1 = [5.0; model.l2+10.0; -5*pi/180.0; -1.0; -1.0; -0.5*pi/180.0]
xT = [0.0; model.l2; 0.0; 0.0; 0.0; 0.0]

# Bounds

# xl <= x <= xl
xl = -Inf*ones(model.nx)
xl[2] = model.l2
xu = Inf*ones(model.nx)

xl_traj = [xl for t = 1:T]
xu_traj = [xu for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT
xl_traj[T][1] = -0.5
xu_traj[T][1] = 0.5


# ul <= u <= uu
uu = [25.0;5.0;10*pi/180.0]
ul = [0.0;-5.0;-10*pi/180.0]

tf0 = 10.0
h0 = tf0/(T-1)
hu = 10*h0
hl = 0*h0

# Objective
Q = [(t != T ? Diagonal([1.0*ones(3);1.0*ones(3)])
    : Diagonal([100.0*ones(3);100.0*ones(3)])) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# Problem
prob = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul for t=1:T-1],
                    uu=[uu for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [1.0e-1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

@show sum(H_nom)
x_pos = [X_nom[t][1] for t = 1:T]
z_pos = [X_nom[t][2] for t = 1:T]

plot(x_pos,z_pos,xlabel="x",ylabel="z",title="Rocket trajectory",
    aspect_ratio=:equal)

plot(x_pos)
plot(z_pos)
plot(hcat(U_nom...)',linetype=:steppost)

# TVLQR policy
Q_lqr = [(t < T ? Diagonal([10.0*ones(3);10.0*ones(3)])
   : Diagonal([100.0*ones(3);100.0*ones(3)])) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [10.0 for t = 1:T-1]

K = TVLQR_gains(model,X_nom,U_nom,H_nom,Q_lqr,R_lqr)




# Simulate policy
using Distributions
model_sim = model_slosh
T_sim = 10*T

# Policy for model with fuel slosh
K_fs = []
X_nom_fs = []
Q_fs = []

for t = 1:T-1
    push!(K_fs,[K[t][:,1:3] zeros(model.nu) K[t][:,4:6] zeros(model.nu)])
end

for t = 1:T
    push!(X_nom_fs,[X_nom[t][1:3];0.0;X_nom[t][4:6];0.0])

    Q_tmp = zeros(model_sim.nx,model_sim.nx)
    Q_tmp[1:3,1:3] = Q[t][1:3,1:3]
    Q_tmp[5:7,5:7] = Q[t][4:6,4:6]
    push!(Q_fs,Q_tmp)
end

x1_sim = [5.0; model.l2+10.0; -5*pi/180.0; 1.0*pi/180.0; -1.0; -1.0; -0.5*pi/180.0; 0.1*pi/180.0]

W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(1.0e-3*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(1.0e-3*ones(model_sim.nx)))
w0 = rand(W0,1)

z1_sim = vec(x1_sim + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K_fs,
    X_nom_fs,U_nom,model_sim,Q_fs,R,T_sim,H_nom[1],z1_sim,w,_norm=2)

plt_x = plot(t_nom,hcat(X_nom_fs...)[1:model_sim.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[1:model_sim.nx,:]',color=:purple,label="",
    width=2.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr
