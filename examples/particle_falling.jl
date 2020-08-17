include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
using Plots

# Horizon
tf = 1.0
T = 101
model.Δt = tf/(T-1)

# Initial and final states
x1 = [0.0; 0.0; 0.5]
xT = [1.0; 0.0; 0.0]

# Bounds
# xl <= x <= xu
xu_traj = [Inf*ones(model.nx) for t=1:T]
xl_traj = [-Inf*ones(model.nx) for t=1:T]

xu_traj[1] = x1
# xu_traj[2] = x1

xl_traj[1] = x1
# xl_traj[2] = x1

xu_traj[T] = xT
xl_traj[T] = xT

# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

# h = h0 (fixed timestep)
hu = model.Δt
hl = model.Δt

# Objective
Q = [t < T ? 0.0*Diagonal(10.0*ones(model.nx)) : 0.0*Diagonal(10.0*ones(model.nx)) for t = 1:T]
R = [0.0*Diagonal(1.0e-1*ones(model.nu_ctrl)) for t = 1:T-2]
c = 0.0

obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])

model.α = 100.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul for t = 1:T-2],
                    uu=[uu for t = 1:T-2],
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2],
					general_constraints=false,
					m_general=0*model.nx,
					general_ineq=(1:0),
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state #TODO clip z
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

x_nom = [X_nom[t][1] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)
plot(x_nom)
plot(z_nom)

model.Δt
X_nom[3]
X_nom[2]
X_nom[1]

t = 5
norm(left_legendre(model,X_nom[t],X_nom[t+1],U_nom[t-1],H_nom[t-2]) - right_legendre(model,X_nom[t-1],X_nom[t],U_nom[t-1],H_nom[t-2]))
