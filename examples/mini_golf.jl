include("../src/sample_trajectory_optimization.jl")
include("../dynamics/golf_particle.jl")
using Plots

# Horizon
tf = 2.0
T = 40
Δt = tf/T

# Initial and final states
x1 = [0.0; 2.0; 0.0]
xT = [0.0; -2.0; 0.0]

# Bounds
# xl <= x <= xu
xu_traj = [Inf*ones(model.nx) for t=1:T]
xl_traj = [-Inf*ones(model.nx) for t=1:T]

xu_traj[1] = x1
xl_traj[1] = x1

xu_traj[2] = x1
xl_traj[2] = x1

# ul <= u <= uu
uu_traj = [[zeros(model.nu_ctrl);Inf*ones(model.nu-model.nu_ctrl)] for t = 1:T-2]
ul_traj = [zeros(model.nu) for t = 1:T-2]

uu_traj[1][1:2] .= Inf
ul_traj[1][1:2] .= -Inf

# h = h0 (fixed timestep)
hu = Δt
hl = Δt

# Objective
Q = [(t<T ? Diagonal(1.0*ones(model.nx))
	: Diagonal(1000.0*ones(model.nx))) for t = 1:T]
R = [Diagonal(1.0e-3*ones(model.nu_ctrl)) for t = 1:T-2]
c = 0.0

x2_ref = range(2.0,stop=-2.0,length=T)
x1_ref = get_y.(x2_ref,2.0,2.0)
x3_ref = exp.(model.a_exp.*(x1_ref .- model.shift_exp))

x_ref = [[x1_ref[t];x2_ref[t];x3_ref[t]] for t = 1:T]

obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])

model.α = 1000.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = deepcopy(x_ref) # linear interpolation on state #TODO clip z
U0 = [0.1*rand(model.nu) for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0),c_tol=1.0e-2,tol=1.0e-2)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

x_nom = [X_nom[t][1] for t = 1:T]
y_nom = [X_nom[t][2] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
b_nom = [U_nom[t][model.idx_b] for t = 1:T-2]
ψ_nom = [U_nom[t][model.idx_ψ[1]] for t = 1:T-2]
η_nom = [U_nom[t][model.idx_η] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)

plot(hcat(x_ref...)[1:1,:]')
plot!(x_nom)
plot(y_nom)
plot(z_nom)
plot(λ_nom)
plot(hcat(b_nom...)')
plot(ψ_nom)
plot(η_nom)
plot(hcat(U_nom...)',linetype=:steppost,label="")

plot(hcat(U_nom...)[1:model.nu_ctrl,:]',linetype=:steppost,label="")

# vis = Visualizer()
# open(vis)
visualize!(vis,model,X_nom)

t = range(-2,stop=2,length=100)
plot(t,exp.(model.a_exp.*(t .- model.shift_exp)),aspect_ratio=:equal)

plot(hcat(x_ref...)[1:3,:]',color=:red,width=2.0,label="")
plot!(hcat(X_nom...)[1:3,:]',color=:black,label="")
