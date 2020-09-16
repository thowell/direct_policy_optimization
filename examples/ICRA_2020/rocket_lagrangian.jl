include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/visualize.jl"))
using Plots, Random

g = 9.81
m1 = 1.0 # mass of rocket
l1 = 0.5 # length from COM to thruster
J = 1.0/12.0*m1*(2*l1)^2 # inertia of rocket

m2 = 0.1 # mass of pendulum
l2 = 0.1 # length from COM to pendulum
l3 = 0.1 # length of pendulum

struct Rocket{T}
	g::T
	m1
	l1
	J

	m2
	l2
	l3

	nx
	nu
end

nx = 8
nu = 2
model_slosh = Rocket(g,m1-m2,l1,J,m2,l2,l3,nx,nu)

struct SimpleRocket{T}
	g::T
	m1
	l1
	J

	nx
	nu
end

nx_simple = 6
nu_simple = 2
model_simple = SimpleRocket(g,m1,l1,J,nx_simple,nu_simple)

function kinematics_thruster(model,q)
	x,z,θ = q[1:3]
	px = x + model.l1*sin(θ)
	pz = z - model.l1*cos(θ)

	return [px;pz]
end

function jacobian_thruster(model::SimpleRocket,q)
	x,z,θ = q[1:3]

	return [1.0 0.0 model.l1*cos(θ);
			0.0 1.0 model.l1*sin(θ)]
end

function jacobian_thruster(model::Rocket,q)
	x,z,θ = q[1:3]

	return [1.0 0.0 model.l1*cos(θ) 0.0;
			0.0 1.0 model.l1*sin(θ) 0.0]
end

function kinematics_mass(model::Rocket,q)
	xp = q[1] + l2*sin(q[3]) + l3*sin(q[4])
	zp = q[2] - l2*cos(q[3]) - l3*cos(q[4])

	return [xp;zp]
end

function jacobian_mass(model::Rocket,q)
	return [1.0 0.0 l2*cos(q[3]) l3*cos(q[4]);
	        0.0 1.0 l2*sin(q[3]) l3*sin(q[4])]
end

function lagrangian(model::SimpleRocket,q,q̇)
	return (0.5*model.m1*(q̇[1]^2 + q̇[2]^2) + 0.5*model.J*q̇[3]^2
			- model.m1*model.g*q[2])
end


function lagrangian(model::Rocket,q,q̇)
	zp = kinematics_mass(model,q)[2]
	vp = jacobian_mass(model,q)*q̇

	return (0.5*model.m1*(q̇[1]^2 + q̇[2]^2) + 0.5*model.J*q̇[3]^2
			- model.m1*model.g*q[2]
			+ 0.5*model.m2*vp'*vp
			- model.m2*model.g*zp)
end

function dLdq(model,q,q̇)
	Lq(x) = lagrangian(model,x,q̇)
	ForwardDiff.gradient(Lq,q)
end

function dLdq̇(model,q,q̇)
	Lq̇(x) = lagrangian(model,q,x)
	ForwardDiff.gradient(Lq̇,q̇)
end

function dynamics(model,x,u)
	nq = convert(Int,floor(model.nx/2))
	q = x[1:nq]
	q̇ = x[nq .+ (1:nq)]
	tmp_q(z) = dLdq̇(model,z,q̇)
	tmp_q̇(z) = dLdq̇(model,q,z)
	[q̇;
	 ForwardDiff.jacobian(tmp_q̇,q̇)\(-1.0*ForwardDiff.jacobian(tmp_q,q)*q̇
	 	+ dLdq(model,q,q̇)
		+ jacobian_thruster(model,q)'*u)]
end

dynamics(model_simple,rand(nx_simple),rand(nu_simple))
dynamics(model_slosh,rand(model_slosh.nx),rand(model_slosh.nu))

vis = Visualizer()
open(vis)

body = Cylinder(Point3f0(0,0,-model.l1),Point3f0(0,0,model.l1),convert(Float32,0.1))
setobject!(vis["rocket"],body,MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

# Model
model = model_simple
nx = model.nx
nu = model.nu

# Horizon
T = 41

# Initial and final states
x1 = [15.0; model.l1+10.0; -45.0*pi/180.0; -10.0; -10.0; -1.0*pi/180.0]
xT = [0.0; model.l1; 0.0; 0.0; 0.0; 0.0]

# Bounds

# xl <= x <= xl
xl = -Inf*ones(model.nx)
xl[2] = model.l1
xu = Inf*ones(model.nx)

xl_traj = [copy(xl) for t = 1:T]
xu_traj = [copy(xu) for t = 1:T]

xl_traj[1] = copy(x1)
xu_traj[1] = copy(x1)

xl_traj[T] = copy(xT)
xu_traj[T] = copy(xT)
xl_traj[T][1] = -0.25
xu_traj[T][1] = 0.25
xl_traj[T][2] = xT[2]-0.001
xu_traj[T][2] = xT[2]+0.001
xl_traj[T][3] = -1.0*pi/180.0
xu_traj[T][3] = 1.0*pi/180.0
xl_traj[T][4] = -0.001
xu_traj[T][4] = 0.001
xl_traj[T][5] = -0.001
xu_traj[T][5] = 0.001
xl_traj[T][6] = -0.01*pi/180.0
xu_traj[T][6] = 0.01*pi/180.0

# ul <= u <= uu
uu = [5.0;100.0]
ul = [-5.0;0.0]

hu = 1.0
hl = 0.01
h0 = 0.5
tf0 = hu*(T-1)

# Objective
Q = [(t != T ? Diagonal([1.0;10.0;1.0;1.0;10.0;1.0])
    : Diagonal([10.0;100.0;10.0;10.0;100.0;10.0])) for t = 1:T]
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
U0 = [rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7,time_limit=20)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

anim = MeshCat.Animation(convert(Int,floor(1/H_nom[1])))
for t = 1:T
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(X_nom[t][1],0.0,X_nom[t][2]),LinearMap(RotY(-1.0*X_nom[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)

# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
plot(hcat(U_nom...)',linetype=:steppost)
sum(H_nom)
H_nom[1]
X_nom[T][2]

# Simulate TVLQR (no slosh)
using Distributions

model_sim = model
x1_sim = [15.0; model.l1+10.0; -45.0*pi/180.0; -10.0; -10.0; -1.0*pi/180.0]

Q_lqr = [(t < T ? Diagonal([100.0*ones(3); 100.0*ones(3)])
   : Diagonal([1000.0*ones(3); 1000.0*ones(3)])) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [100.0 for t = 1:T-1]

K = TVLQR_gains(model,X_nom,U_nom,[H_nom[1] for t = 1:T-1],Q_lqr,R_lqr)

T_sim = 10*T

W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(1.0e-5*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(1.0e-5*ones(model_sim.nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(x1) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[1:model.nx,:]',color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],color=:black,
    label="",width=1.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr

plot(hcat(U_nom...)')

# Simulate TVLQR with fuel slosh

x1_slosh = [15.0; model.l1+10.0; -45.0*pi/180.0; 0.0; -10.0; -10.0; -1.0*pi/180.0; 0.0]
xT_slosh = [xT[1:3];0.0;xT[4:6]...;0.0]

function policy(model::Rocket,K,x,u,x_nom,u_nom)
	u_nom - reshape(K,model.nu,model.nx-2)*(output(model,x) - x_nom)
end

function output(model::Rocket,x)
	x[[(1:3)...,(5:7)...]]
end

# optimize slosh model

# xl <= x <= xl
xl_slosh = -Inf*ones(model_slosh.nx)
xl_slosh[2] = model_slosh.l1
xu_slosh = Inf*ones(model_slosh.nx)

xl_traj_slosh = [copy(xl_slosh) for t = 1:T]
xu_traj_slosh = [copy(xu_slosh) for t = 1:T]

xl_traj_slosh[1] = copy(x1_slosh)
xu_traj_slosh[1] = copy(x1_slosh)

xl_traj_slosh[T] = copy(xT_slosh)
xu_traj_slosh[T] = copy(xT_slosh)

xl_traj_slosh[T][4] = -Inf
xu_traj_slosh[T][4] = Inf

xl_traj_slosh[T][8] = -Inf
xu_traj_slosh[T][8] = Inf

xl_traj_slosh[T][1] = -0.25
xu_traj_slosh[T][1] = 0.25
xl_traj_slosh[T][2] = xT[2]-0.001
xu_traj_slosh[T][2] = xT[2]+0.001
xl_traj_slosh[T][3] = -1.0*pi/180.0
xu_traj_slosh[T][3] = 1.0*pi/180.0
xl_traj_slosh[T][5] = -0.001
xu_traj_slosh[T][5] = 0.001
xl_traj_slosh[T][6] = -0.001
xu_traj_slosh[T][6] = 0.001
xl_traj_slosh[T][7] = -0.01*pi/180.0
xu_traj_slosh[T][7] = 0.01*pi/180.0

Q_slosh = [(t != T ? Diagonal([1.0;10.0;1.0;0.1;1.0;10.0;1.0;0.1])
    : Diagonal([10.0;100.0;10.0;0.1;10.0;100.0;10.0;0.1])) for t = 1:T]
obj_slosh = QuadraticTrackingObjective(Q_slosh,R,c,
    [xT_slosh for t=1:T],[zeros(model.nu) for t=1:T])

# Problem
prob_slosh = init_problem(model_slosh.nx,model_slosh.nu,T,model_slosh,obj_slosh,
                    xl=xl_traj_slosh,
                    xu=xu_traj_slosh,
                    ul=[ul for t=1:T-1],
                    uu=[uu for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi_slosh = init_MOI_Problem(prob_slosh)

# Trajectory initialization
X0_slosh = linear_interp(x1_slosh,xT_slosh,T) # linear interpolation on state
U0_slosh = [rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0_slosh = pack(X0_slosh,U0_slosh,h0,prob_slosh)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal_slosh = solve(prob_moi_slosh,copy(Z0_slosh),nlp=:SNOPT7,time_limit=20)
X_nom_slosh, U_nom_slosh, H_nom_slosh = unpack(Z_nominal_slosh,prob_slosh)

anim = MeshCat.Animation(convert(Int,floor(1/H_nom_slosh[1])))
for t = 1:T
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(X_nom_slosh[t][1],0.0,X_nom_slosh[t][2]),LinearMap(RotY(-1.0*X_nom_slosh[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)
# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
plot(hcat(U_nom_slosh...)',linetype=:steppost)
sum(H_nom_slosh)
sum(H_nom)
X_nom_slosh[T][2]

# simulate slosh with TVLQR controller from nominal model
model_sim = model_slosh
W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w0 = rand(W0,1)

x1_slosh_sim = [15.0; model.l1+10.0; -45.0*pi/180.0;0.0; -10.0; -10.0; -1.0*pi/180.0;0.0]

z0_sim = vec(copy(x1_slosh_sim) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)
dt_sim = sum(H_nom)/(T_sim-1)
z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[[(1:3)...,(5:7)...],:]',color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],color=:black,
    label="",width=1.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr

anim = MeshCat.Animation(convert(Int,floor(1/dt_sim)))
for t = 1:T_sim
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(z_tvlqr[t][1],0.0,z_tvlqr[t][2]),LinearMap(RotY(-1.0*z_tvlqr[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)

# DPO
N = 2*model_slosh.nx
# mf = range(0.9*model_slosh.mf,stop=1.1*model_slosh.mf,length=N)
# mr = [1.0-mf[i] for i = 1:N]
# lf = shuffle(range(0.9*model_slosh.l3,stop=1.1*model_slosh.l3,length=N))
# models = [RocketSlosh(mr[i],model_slosh.Jr,mf[i],model_slosh.g,model_slosh.l1,model_slosh.l2,
#     lf[i],model_slosh.nx,model_slosh.nu) for i = 1:N]

models = [model_slosh for i = 1:N]
# ψ = shuffle(range(-pi/2,stop=pi/2,length=N))
# dψ = shuffle(range(-pi/10,stop=pi/10,length=N))

β = 1.0
w = 1.0e-3*ones(model_slosh.nx)
γ = N
x1_sample = resample([x1_slosh for i = 1:N],β=β,w=w)
# x1_sample = [x1_slosh for i = 1:N]
#
# for i = 1:N
# 	x1_sample[i][4] = ψ[i]
# 	x1_sample[i][8] = dψ[i]
# end

xl_traj_sample = [[-Inf*ones(model_slosh.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model_slosh.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = copy(x1_sample[i])
    xu_traj_sample[i][1] = copy(x1_sample[i])

	# xl_traj_sample[i][T] = copy(xT_slosh)
	# xu_traj_sample[i][T] = copy(xT_slosh)

	# xl_traj_sample[i][T][4] = -Inf
	# xu_traj_sample[i][T][4] = Inf
	#
	# xl_traj_sample[i][T][8] = -Inf
	# xu_traj_sample[i][T][8] = Inf
	#
	# xl_traj_sample[i][T][1] = -0.25
	# xu_traj_sample[i][T][1] = 0.25
	# xl_traj_sample[i][T][2] = xT[2]-0.01
	# xu_traj_sample[i][T][2] = xT[2]+0.01
	# xl_traj_sample[i][T][3] = -1.0*pi/180.0
	# xu_traj_sample[i][T][3] = 1.0*pi/180.0
	# xl_traj_sample[i][T][5] = -0.001
	# xu_traj_sample[i][T][5] = 0.001
	# xl_traj_sample[i][T][6] = -0.001
	# xu_traj_sample[i][T][6] = 0.001
	# xl_traj_sample[i][T][7] = -0.01*pi/180.0
	# xu_traj_sample[i][T][7] = 0.01*pi/180.0
end

ul_traj_sample = [[-Inf*ones(model_slosh.nu) for t = 1:T-1] for i = 1:N]
uu_traj_sample = [[Inf*ones(model_slosh.nu) for t = 1:T-1] for i = 1:N]

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
	# ul=ul_traj_sample,
    # uu=uu_traj_sample,
    n_features=model_slosh.nx-2,
	policy_constraint=true
    )

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = zeros(prob_sample_moi.n)

Z0_sample[prob_sample.idx_nom_z] = pack(X_nom,U_nom,H_nom[1],prob)

for t = 1:T
	for i = 1:N
		Z0_sample[prob_sample.idx_sample[i].x[t]] = X_nom_slosh[t]
		t==T && continue
		Z0_sample[prob_sample.idx_sample[i].u[t]] = U_nom_slosh[t]
		Z0_sample[prob_sample.idx_sample[i].h[t]] = H_nom_slosh[1]
		Z0_sample[prob_sample.idx_x_tmp[i].x[t]] = X_nom_slosh[t+1]
	end
end

for t = 1:T-1
	Z0_sample[prob_sample.idx_K[t]] = vec(K[t])
end

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),max_iter=500,nlp=:SNOPT7,time_limit=60*5,tol=1.0e-2,c_tol=1.0e-2)

X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
sum(H_nom_sample)
sum(H_nom)

# prob_sample2 = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
#     xl=xl_traj_sample,
#     xu=xu_traj_sample,
#     n_features=model_slosh.nx-2,
# 	policy_constraint=false
#     )
# prob_sample_moi2 = init_MOI_Problem(prob_sample2)
#
# X_tmp = []
# for t = 1:T
# 	push!(X_tmp,[X_nom_sample[t][1:3]...;0.0;X_nom_sample[t][4:6]...;0.0])
# end
# Z0_sample2 = pack(X_nom_sample,X_tmp,U_nom_sample,H_nom_sample[1],[vec(K[t]) for t = 1:T-1],prob_sample2)
#
# Z_sample_sol = solve(prob_sample_moi2,copy(Z0_sample2),max_iter=500,nlp=:SNOPT7,time_limit=60*5,tol=1.0e-2,c_tol=1.0e-2)
# X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample2)
# Θ = [Z_sample_sol[prob_sample2.idx_K[t]] for t = 1:T-1]

Θ = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]

z0_sim = vec(copy(x1_slosh_sim) + w0)

t_nom_sample = range(0,stop=sum(H_nom_sample),length=T)
t_sim_nom_sample = range(0,stop=sum(H_nom_sample),length=T_sim)

dt_sim_nom = sum(H_nom)/(T_sim-1)
dt_sim_sample = sum(H_nom_sample)/(T_sim-1)

W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w0 = rand(W0,1)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[[(1:3)...,(5:7)...],:]',color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],color=:black,
    label="",width=1.0)

z_sample, u_sample, J_sample, Jx_sample, Ju_sample= simulate_linear_controller(Θ,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[1:model.nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample...)[[(1:3)...,(5:7)...],:]',color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_sample...)[1,:],hcat(z_sample...)[2,:],color=:black,
    label="",width=1.0)

# objective value
J_sample
J_tvlqr

# state tracking
Jx_sample
Jx_tvlqr

# control tracking
Ju_sample
Ju_tvlqr

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],legend=:topright,color=:purple,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")

plot_traj = plot!(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],legend=:topright,color=:orange,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")

for i = 1:N
	plot_traj = plot!(hcat(X_sample[i]...)[1,:],hcat(X_sample[i]...)[2,:],label="")
end
display(plot_traj)

# orientation tracking
plt_x = plot(t_nom,hcat(X_nom...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample...)[3,:],color=:black,label="",
    width=1.0)

sum(H_nom_sample)
sum(H_nom)

using PGFPlots
const PGF = PGFPlots

p_traj_nom = PGF.Plots.Linear(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	mark="",style="color=cyan, line width = 2pt",legendentry="TO")
p_traj_sample = PGF.Plots.Linear(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],
	mark="",style="color=orange, line width = 2pt",legendentry="DPO")

a = Axis([p_traj_nom;p_traj_sample],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="z",
	xlabel="y",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"rocket_traj.tikz"), a, include_preamble=false)


# orientation tracking
plt_x = plot(t_nom,hcat(X_nom...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[3,:],color=:black,label="",
    width=1.0)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[3,:],legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom_sample,hcat(z_sample...)[3,:],color=:black,label="",
    width=1.0)


p_nom_orientation = PGF.Plots.Linear(t_nom,hcat(X_nom...)[3,:],
	mark="",style="color=cyan, line width = 2pt",legendentry="TO")
p_nom_sim_orientation = PGF.Plots.Linear(t_sim_nom,hcat(z_tvlqr...)[3,:],
	mark="",style="color=black, line width = 1pt")

p_sample_orientation = PGF.Plots.Linear(t_nom_sample,hcat(X_nom_sample...)[3,:],
	mark="",style="color=orange, line width = 2pt",legendentry="DPO")
p_sample_sim_orientation = PGF.Plots.Linear(t_sim_nom_sample,hcat(z_sample...)[3,:],
	mark="",style="color=black, line width = 1pt")

a = Axis([p_nom_orientation;p_nom_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"rocket_tvlqr_orientation.tikz"), a, include_preamble=false)

a = Axis([p_sample_orientation;p_sample_sim_orientation],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="orientation",
	xlabel="time",
	legendStyle="{at={(0.5,0.99)},anchor=north}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"rocket_dpo_orientation.tikz"), a, include_preamble=false)


anim = MeshCat.Animation(convert(Int,floor(1/dt_sim_nom)))
for t = 1:T_sim
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(z_tvlqr[t][1],0.0,z_tvlqr[t][2]),LinearMap(RotY(-1.0*z_tvlqr[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)

anim = MeshCat.Animation(convert(Int,floor(1/dt_sim_sample)))
for t = 1:T_sim
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(z_sample[t][1],0.0,z_sample[t][2]),LinearMap(RotY(-1.0*z_sample[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)


landing_pad = Cylinder(Point3f0(0,0,-0.1),Point3f0(0,0,0),convert(Float32,0.25))
setobject!(vis["landing_pad"],landing_pad,MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
