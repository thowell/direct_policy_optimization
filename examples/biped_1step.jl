include("../src/sample_trajectory_optimization.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 10
Tm = -1
model.Tm = Tm

# Initial and final states
q_init = [2.44917526290273,2.78819438807838,0.812693850088907,0.951793806080012,1.49974183648719e-13]
v_init = [0.0328917694318260,0.656277832705193,0.0441573173000750,1.03766701983449,1.39626340159558]
q_final = [2.84353387648829,2.48652580597252,0.751072212241267,0.645830978766432, 0.0612754113212848]
v_final = [2.47750069399969,3.99102008940145,-1.91724136219709,-3.95094757056324,0.0492401787458546]

x1 = [q_init;v_init]
# xT = [q_final;v_final]
xT = Δ(x1)

# Bounds

# xl <= x <= xu
xl_traj = [t != T ? -Inf*ones(model.nx) : xT for t = 1:T]
xu_traj = [t != T ? Inf*ones(model.nx) : xT for t = 1:T]

# ul <= u <= uu
uu = 1000.0
ul = -1000.0

tf0 = 0.36915
h0 = tf0/(T-1)
hu = h0
hl = h0

function discrete_dynamics(model::Biped,x⁺,x,u,h,t)
    if t == model.Tm
        rk3_implicit(model,x⁺,Δ(x),u,h)
    else
        return rk3_implicit(model,x⁺,x,u,h)
    end
end

# pfz_init = kinematics(model,q_init)[2]
#
# function c_stage!(c,x,u,t,model)
#     c[1] = kinematics(model,x[1:5])[2] - pfz_init
#     nothing
# end
#
# m_stage = 1

# Objective
Q = [t < T ? Diagonal([1.0e-3*ones(5);1.0e-3*ones(5)]) : Diagonal(1.0e-3*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-3*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1.0,0.05,[t for t = 1:T-1 if (t != Tm-1 || t != 1)])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true,
                    stage_constraints=false)#,
                    # m_stage=[t==Tm ? 0 : m_stage for t=1:T-1]
                    # )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

Q_nominal = [X_nominal[t][1:5] for t = 1:T]

foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=1:T]
foot_y = [foot_traj[t][2] for t=1:T]

plt1 = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="")

# TVLQR policy
Q_lqr = [t < T ? Diagonal(1.0*ones(model.nx)) : Diagonal(1.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(0.01*ones(model.nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

A = []
B = []
for t = 1:T-1
    x = X_nominal[t]
    u = U_nominal[t]
    h = H_nominal[t]
    x⁺ = X_nominal[t+1]

    fx(z) = discrete_dynamics(model,x⁺,z,u,h,t)
    fu(z) = discrete_dynamics(model,x⁺,x,z,h,t)
    fx⁺(z) = discrete_dynamics(model,z,x,u,h,t)

    A⁺ = ForwardDiff.jacobian(fx⁺,x⁺)
    push!(A,-A⁺\ForwardDiff.jacobian(fx,x))
    push!(B,-A⁺\ForwardDiff.jacobian(fu,u))
end

K = TVLQR(A,B,Q_lqr,R_lqr)

# Samples
N = 2*model.nx
models = [model for i = 1:N]
# K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 1.0e-8*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    disturbance_ctrl=false,α=1000.0)
prob_sample_moi = init_MOI_Problem(prob_sample)


# for i = 1:N
#     prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[T][6:10]] .= -Inf
#     prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[T][6:10]] .= Inf
# end

# for i = 1:N
#     prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[T]] .= -Inf
#     prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[T]] .= Inf
# end

# for i = 1:N
#     prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[models[i].Tm]] .= [0.0;-Inf]
#     prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[models[i].Tm]] .= [0.0;Inf]
#     # prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[models[i].Tm+1]] .= [1.0;-Inf]
#     # prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[models[i].Tm+1]] .= [1.0;Inf]
#
#     # prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[T] .= [1.0;-Inf]
#     # prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[T]] .= [1.0;Inf]
# end

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)


Q_nom_sample = [X_nom_sample[t][1:5] for t = 1:T]

foot_traj_nom_sample = [kinematics(model,Q_nom_sample[t]) for t = 1:T]

foot_x_ns = [foot_traj_nom_sample[t][1] for t=1:T]
foot_y_ns = [foot_traj_nom_sample[t][2] for t=1:T]

plt1 = plot(foot_x_ns,foot_y_ns,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="")


for i = 1:N
    Q_sample = [X_sample[i][t][1:5] for t = 1:T]

    foot_traj_sample = [kinematics(model,Q_sample[t]) for t = 1:T]

    foot_x_s = [foot_traj_sample[t][1] for t=1:T]
    foot_y_s = [foot_traj_sample[t][2] for t=1:T]

    plt1 = plot!(foot_x_s,foot_y_s,aspect_ratio=:equal,label="")
end
display(plt1)
