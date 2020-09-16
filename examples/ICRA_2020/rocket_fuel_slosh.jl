include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/rocket.jl"))
include(joinpath(pwd(),"dynamics/visualize.jl"))
using Plots, Random

vis = Visualizer()
open(vis)

# Nominal model
nx_nom = model_nom.nx
nu = model_nom.nu
nq_nom = convert(Int,floor(nx_nom/2))

# radius of landing pad
r_pad = 0.25

# Horizon
T = 41

# Initial and final states
x1 = [15.0; model_nom.l1+10.0; -45.0*pi/180.0; -10.0; -10.0; -1.0*pi/180.0]
xT = [0.0; model_nom.l1; 0.0; 0.0; 0.0; 0.0]

# Bounds

# xl <= x <= xl
xl = -Inf*ones(model_nom.nx)
xl[2] = model_nom.l1
xu = Inf*ones(model_nom.nx)

xl_traj = [copy(xl) for t = 1:T]
xu_traj = [copy(xu) for t = 1:T]

xl_traj[1] = copy(x1)
xu_traj[1] = copy(x1)

xl_traj[T] = copy(xT)
xu_traj[T] = copy(xT)
xl_traj[T][1] = -r_pad
xu_traj[T][1] = r_pad
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
R = [Diagonal(1.0e-1*ones(model_nom.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model_nom.nu) for t=1:T])

# Problem
prob_nom = init_problem(model_nom.nx,model_nom.nu,T,model_nom,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul for t=1:T-1],
                    uu=[uu for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    )

# MathOptInterface problem
prob_nom_moi = init_MOI_Problem(prob_nom)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [rand(model_nom.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob_nom)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nom = solve(prob_nom_moi,copy(Z0),nlp=:SNOPT7,time_limit=20)
X_nom, U_nom, H_nom = unpack(Z_nom,prob_nom)

visualize!(vis,model_nom,X_nom,Δt=H_nom[1],
	r_pad=r_pad)

plot(hcat(U_nom...)',linetype=:steppost)
sum(H_nom) # works when 2.72

# TVLQR policy
Q_lqr = [(t < T ? Diagonal([100.0*ones(nq_nom); 100.0*ones(nq_nom)])
   : Diagonal([1000.0*ones(nq_nom); 1000.0*ones(nq_nom)])) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model_nom.nu)) for t = 1:T-1]
H_lqr = [100.0 for t = 1:T-1]

K = TVLQR_gains(model_nom,X_nom,U_nom,[H_nom[1] for t = 1:T-1],Q_lqr,R_lqr)

# Simulate TVLQR (no slosh)
using Distributions

model_sim = model_nom
x1_sim = copy(x1)
T_sim = 10*T

W = Distributions.MvNormal(zeros(model_sim.nx),
	Diagonal(1.0e-5*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),
	Diagonal(1.0e-5*ones(model_sim.nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(x1_sim) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)

# simulate
z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	ul=ul,uu=uu)

# plot states
plt_x = plot(t_nom,hcat(X_nom...)[1:model_nom.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[1:model_nom.nx,:]',color=:black,label="",
    width=1.0)

# plot COM
plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],
	color=:black,
    label="",width=1.0)

plot(t_nom[1:end-1],hcat(U_nom...)',
	linetype=:steppost,color=:red,width=2.0)
plot!(t_sim_nom[1:end-1],hcat(u_tvlqr...)',
	linetype=:steppost,color=:black,width=1.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr


# Fuel-slosh model
x1_slosh = [x1[1:3];0.0;x1[4:6]...;0.0]
xT_slosh = [xT[1:3];0.0;xT[4:6]...;0.0]

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

xl_traj_slosh[T][1] = -r_pad
xu_traj_slosh[T][1] = r_pad
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
    [xT_slosh for t=1:T],[zeros(model_slosh.nu) for t=1:T])

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
U0_slosh = [rand(model_slosh.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0_slosh = pack(X0_slosh,U0_slosh,h0,prob_slosh)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal_slosh = solve(prob_moi_slosh,copy(Z0_slosh),nlp=:SNOPT7,time_limit=20)
X_nom_slosh, U_nom_slosh, H_nom_slosh = unpack(Z_nominal_slosh,prob_slosh)

plot(hcat(U_nom_slosh...)',linetype=:steppost)
sum(H_nom_slosh) # should be 2.76
sum(H_nom) # should be 2.72
X_nom_slosh[T][2]

# simulate slosh with TVLQR controller from nominal model
model_sim = model_slosh
W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w0 = rand(W0,1)

x1_slosh_sim = x1_slosh
z0_sim = vec(copy(x1_slosh_sim) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)
dt_sim_nom = sum(H_nom)/(T_sim-1)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model_nom.nx,:]',
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[[(1:3)...,(5:7)...],:]',
	color=:black,label="",
    width=1.0)

plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
	legend=:topright,color=:red,
    label="",width=2.0,xlabel="y",ylabel="z",
    title="Rocket")
plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],
	color=:black,
    label="",width=1.0)

# objective value
J_tvlqr

# state tracking
Jx_tvlqr

# control tracking
Ju_tvlqr

visualize!(vis,model_slosh,z_tvlqr,Δt=dt_sim_nom,
	r_pad=r_pad)

# DPO
N = 2*model_slosh.nx
models = [model_slosh for i = 1:N]
β = 1.0
w = 1.0e-3*ones(model_slosh.nx)
γ = N

x1_sample = resample([x1_slosh for i = 1:N],β=β,w=w)

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

prob_sample = init_sample_problem(prob_nom,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
	# ul=ul_traj_sample,
    # uu=uu_traj_sample,
    n_features=model_slosh.nx-2,
	policy_constraint=true
    )

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = zeros(prob_sample_moi.n)

Z0_sample[prob_sample.idx_nom_z] = pack(X_nom,U_nom,H_nom[1],prob_nom)

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
sum(H_nom_sample) # should be 2.9
sum(H_nom)

# get policy
Θ = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]

t_nom_sample = range(0,stop=sum(H_nom_sample),length=T)
t_sim_nom_sample = range(0,stop=sum(H_nom_sample),length=T_sim)

dt_sim_sample = sum(H_nom_sample)/(T_sim-1)

W = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.nx),Diagonal(0.0*ones(model_sim.nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(x1_slosh_sim) + w0)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom,hcat(X_nom...)[1:model_nom.nx,:]',legend=:topright,color=:red,
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
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],z0_sim,
	w,_norm=2,
	controller=:policy,ul=ul,uu=uu)

plt_x = plot(t_nom_sample,hcat(X_nom_sample...)[1:model_nom.nx,:]',legend=:topright,color=:red,
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

visualize!(vis,model_nom,z_tvlqr,Δt=dt_sim_nom,
	r_pad=r_pad)

visualize!(vis,model_nom,z_sample,Δt=dt_sim_sample,
	r_pad=r_pad)

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
