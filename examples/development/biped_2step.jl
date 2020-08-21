include("../src/sample_motion_planning.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 20
Tm = 10
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
xl_traj = [t != Tm ? -Inf*ones(model.nx) : [xT[1:5];-Inf*ones(5)] for t = 1:T]
xu_traj = [t != Tm ? Inf*ones(model.nx) : [xT[1:5];Inf*ones(5)] for t = 1:T]

# ul <= u <= uu
uu = 20.0
ul = -20.0

tf0 = 2.0*0.36915
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
Q = [t < T ? Diagonal(1.0e-3*ones(model.nx)) : Diagonal(1.0e-3*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-3*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1.0,0.05,[t for t = 1:T-1 if (t != Tm-1 || t != 1)])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
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

norm(vec(X_nominal[Tm][1:5] - X_nominal[T][1:5]))
Q_nominal = [X_nominal[t][1:5] for t = 1:T]

foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=1:Tm]
foot_y = [foot_traj[t][2] for t=1:Tm]

plt1 = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="")

foot_x = [foot_traj[t][1] for t=Tm+1:T]
foot_y = [foot_traj[t][2] for t=Tm+1:T]
foot_tran = kinematics(model,Δ(X_nominal[Tm])[1:5])

plt1 = plot!([foot_tran[1],foot_x...],[foot_tran[2],foot_y...],aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
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
w = 1.0e-5*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)
xT_sample = resample([xT for i = 1:N],β=β,w=w)

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    disturbance_ctrl=true,α=1.0e-5)
prob_sample_moi = init_MOI_Problem(prob_sample)

for i = 1:N
    prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[Tm]] .= [xT_sample[i][1:5];-Inf*ones(5)]
    prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[Tm]] .= [xT_sample[i][1:5];Inf*ones(5)]
end

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=500)
# Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,max_iter=100)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
Uw_sample = unpack_disturbance(Z_sample_sol,prob_sample)

Q_nom_sample = [X_nom_sample[t][1:5] for t = 1:T]

foot_traj_nom_sample = [kinematics(model,Q_nom_sample[t]) for t = 1:T]

foot_x_ns = [foot_traj_nom_sample[t][1] for t=1:T]
foot_y_ns = [foot_traj_nom_sample[t][2] for t=1:T]

plt1 = plot(aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="")

for i = 1:N
    Q_sample = [X_sample[i][t][1:5] for t = 1:T]

    foot_traj_sample = [kinematics(model,Q_sample[t]) for t = 1:Tm]

    foot_x_s = [foot_traj_sample[t][1] for t=1:Tm]
    foot_y_s = [foot_traj_sample[t][2] for t=1:Tm]

    plt1 = plot!(foot_x_s,foot_y_s,aspect_ratio=:equal,label="")
end
plt1 = plot!(foot_x_ns[1:Tm],foot_y_ns[1:Tm],aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",color=:red,label="nominal",legend=:bottom)
display(plt1)

foot_x_ns = [foot_traj_nom_sample[t][1] for t=Tm+1:T]
foot_y_ns = [foot_traj_nom_sample[t][2] for t=Tm+1:T]

plt2 = plot(aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 2 trajectory",label="")

for i = 1:N
    Q_sample = [X_sample[i][t][1:5] for t = 1:T]

    foot_traj_sample = [kinematics(model,Q_sample[t]) for t = 1:T]

    foot_x_s = [foot_traj_sample[t][1] for t=Tm+1:T]
    foot_y_s = [foot_traj_sample[t][2] for t=Tm+1:T]

    foot_tran = kinematics(model,Δ(X_sample[i][Tm])[1:5])

    plt2 = plot!([foot_tran[1],foot_x_s...],[foot_tran[2],foot_y_s...],aspect_ratio=:equal,label="")
end
foot_tran_ns = kinematics(model,Δ(X_nom_sample[Tm])[1:5])

plt2 = plot!([foot_tran_ns[1],foot_x_ns...],[foot_tran_ns[2],foot_y_ns...],aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 2 trajectory",color=:red,label="nominal",legend=:bottom)
display(plt2)

plot(plt1,plt2,layout=(2,1))

plt = plot()
for i = 1:N
    plt = plot!(hcat(Uw_sample[i]...)[1:end,:]',linetype=:steppost,labels="")
end
display(plt)

t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

# State samples
plt1 = plot();
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1:5,:]',label="",linetype=:steppost);
end
plt1 = plot!(t_sample,hcat(X_nom_sample...)[1:5,:]',color=:red,width=2.0,
    label="",linetype=:steppost,title="Biped state");
display(plt1)
# savefig(plt,joinpath(@__DIR__,"results/double_integrator_sample_state.png"))

# Control samples
plt3 = plot();
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1:4,:]',label="",
        linetype=:steppost);
end

plt3 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)[1:4,:]',color=:red,width=2.0,
    title="Biped control",label="",xlabel="time (s)",linetype=:steppost);
display(plt3)

K_sample = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]
norm(vec(hcat(K...)) - vec(hcat(K_sample...)))
