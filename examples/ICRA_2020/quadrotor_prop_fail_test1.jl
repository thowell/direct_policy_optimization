include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/quadrotor.jl"))
# include(joinpath(pwd(),"dynamics/obstacles.jl"))
# include(joinpath(pwd(),"dynamics/visualize.jl"))

# vis = Visualizer()
# open(vis)

# using Plots

# Horizon
T = 31
# Tm = convert(Int,floor(T/2)+1)

# Bounds

# ul <= u <= uu
uu = 3.0*ones(model.nu)
uu_nom = copy(uu)
# uu[2] *= 0.5
ul = zeros(model.nu)
ul_nom = copy(ul)
@assert sum(uu) > -1.0*model.m*model.g[3]

uu_traj = [copy(uu_nom) for t = 1:T-1]
ul_traj = [copy(ul_nom) for t = 1:T-1]

# for t = Tm:T-1
#     uu_traj[t][1] = 0.0
# end

# Circle obstacle
# r_cyl = 0.5
# r = r_cyl + model.L
# xc1 = 1.5-0.1
# yc1 = 1.5
# xc2 = 4.0-0.125
# yc2 = 4.0
# xc3 = 2.25
# yc3 = 1.125
# xc4 = 2.0
# yc4 = 4.0

# xc = [xc1,xc2,xc3,xc4]
# yc = [yc1,yc2,yc3,yc4]
#
# circles = [(xc1,yc1,r),(xc2,yc2,r),(xc3,yc3,r),(xc4,yc4,r)]
#
# # Constraints
# function c_stage!(c,x,u,t,model)
#     c[1] = circle_obs(x[1],x[2],xc1,yc1,r)
#     # c[2] = circle_obs(x[1],x[2],xc2,yc2,r)
#     # c[3] = circle_obs(x[1],x[2],xc3,yc3,r)
#     # c[4] = circle_obs(x[1],x[2],xc4,yc4,r)
#     nothing
# end
# m_stage = 1

# h = h0 (fixed timestep)
tf0 = 5.0
h0 = tf0/(T-1)
hu = h0
hl = 0.0*h0

# Initial and final states
x1 = zeros(model.nx)
x1[3] = 1.0
xT = copy(x1)
xT[1] = 3.0
xT[2] = 3.0

xl = -Inf*ones(model.nx)
# xl[1] = -1.0
# xl[2] = -1.0
xl[3] = 0.0

xu = Inf*ones(model.nx)
# xu[1] = 6.0
# xu[2] = 6.0
xl_traj = [copy(xl) for t = 1:T]
xu_traj = [copy(xu) for t = 1:T]

xl_traj[1] = copy(x1)
xu_traj[1] = copy(x1)

xl_traj[T] = copy(xT)
xu_traj[T] = copy(xT)

u_ref = -1.0*model.m*model.g[3]/4.0*ones(model.nu)

# Objective
Q = [t < T ? Diagonal(ones(model.nx)) : Diagonal(1.0*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 100.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[u_ref for t=1:T-1])

# TVLQR cost
Q_lqr = [t < T ? Diagonal(10.0*ones(model.nx)) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [1.0 for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    # stage_constraints=true,
                    # m_stage=[m_stage for t=1:T-1]
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [copy(u_ref) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7,c_tol=1.0e-3,tol=1.0e-3)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
sum(H_nom)
# plot(hcat(X_nom...)[1:3,:]',linetype=:steppost)
# plot(hcat(U_nom...)',linetype=:steppost)

# visualize!(vis,model,X_nom,Δt=H_nom[1])
# for i = 1:m_stage
#     cyl = Cylinder(Point3f0(xc[i],yc[i],0),Point3f0(xc[i],yc[i],2.0),convert(Float32,r_cyl-model.L))
#     setobject!(vis["cyl$i"],cyl,MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
# end

# Simulate TVLQR
K = TVLQR_gains(model,X_nom,U_nom,[H_nom[1] for t = 1:T-1],Q_lqr,R_lqr)

# using Distributions
#
# model_sim = model
# x1_sim = copy(x1)
# T_sim = 10*T
#
# W = Distributions.MvNormal(zeros(model_sim.nx),
# 	Diagonal(1.0e-32*ones(model_sim.nx)))
# w = rand(W,T_sim)
#
# W0 = Distributions.MvNormal(zeros(model_sim.nx),
# 	Diagonal(1.0e-32*ones(model_sim.nx)))
# w0 = rand(W0,1)
#
# z0_sim = vec(copy(x1_sim) + w0)
#
# t_nom = range(0,stop=sum(H_nom),length=T)
# t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)
#
# # simulate
# z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
#     X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,H_nom[1],z0_sim,w,_norm=2,
# 	ul=ul_nom,uu=uu_nom)
#
# # plot states
# plt_x = plot(t_nom,hcat(X_nom...)[1:model.nx,:]',
# 	legend=:topright,color=:red,
#     label="",width=2.0,xlabel="time (s)",
#     title="Quadrotor",ylabel="state")
# plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[1:model.nx,:]',color=:black,label="",
#     width=1.0)
#
# # plot COM
# plot_traj = plot(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
# 	legend=:topright,color=:red,
#     label="",width=2.0,xlabel="y",ylabel="z",
#     title="Quadrotor")
# plot_traj = plot!(hcat(z_tvlqr...)[1,:],hcat(z_tvlqr...)[2,:],
# 	color=:black,
#     label="",width=1.0)
#
# plot(t_nom[1:end-1],hcat(U_nom...)',
# 	linetype=:steppost,color=:red,width=2.0)
# plot!(t_sim_nom[1:end-1],hcat(u_tvlqr...)',
# 	linetype=:steppost,color=:black,width=1.0)

# Sample
N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-3*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=1.0,w=1.0e-3*ones(model.nx))

xl_traj_sample = [[copy(xl) for t = 1:T] for i = 1:N]
xu_traj_sample = [[copy(xu) for t = 1:T] for i = 1:N]

ul_traj_sample = [[copy(ul_nom) for t = 1:T-1] for i = 1:N]
uu_traj_sample = [[copy(uu_nom) for t = 1:T-1] for i = 1:N]
for t = 1:T-1
	for i = 1:6
		uu_traj_sample[i][t][1] *= 0.5
	end
	for i = 6 .+ (1:6)
		uu_traj_sample[i][t][2] *= 0.5
	end
	for i = 12 .+ (1:6)
		uu_traj_sample[i][t][3] *= 0.5
	end
	for i = 18 .+ (1:6)
		uu_traj_sample[i][t][4] *= 0.5
	end
end

for i = 1:N
    xl_traj_sample[i][1] = copy(x1_sample[i])
    xu_traj_sample[i][1] = copy(x1_sample[i])
end

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
	ul=ul_traj_sample,
    uu=uu_traj_sample,
    β=β,w=w,γ=γ)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K,prob_sample,r=0.001)


# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7,
    time_limit=60,tol=1.0e-2,c_tol=1.0e-2)

using JLD
@save joinpath(pwd(),"res.jld") Z_sample_sol

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

# Time trajectories
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nom))s")
display("time (sample): $(sum(H_nom_sample))s")

# # Plots results
#
# # Position trajectory
# x_nom_pos = [X_nom[t][1] for t = 1:T]
# y_nom_pos = [X_nom[t][2] for t = 1:T]
# pts = Plots.partialcircle(0,2π,100,r)
# cx,cy = Plots.unzip(pts)
# cx1 = [_cx + xc1 for _cx in cx]
# cy1 = [_cy + yc1 for _cy in cy]
# cx2 = [_cx + xc2 for _cx in cx]
# cy2 = [_cy + yc2 for _cy in cy]
# cx3 = [_cx + xc3 for _cx in cx]
# cy3 = [_cy + yc3 for _cy in cy]
# cx4 = [_cx + xc4 for _cx in cx]
# cy4 = [_cy + yc4 for _cy in cy]
# # cx5 = [_cx + xc5 for _cx in cx]
# # cy5 = [_cy + yc5 for _cy in cy]
#
# plt = plot(Shape(cx1,cy1),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx2,cy2),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx3,cy3),color=:red,label="",linecolor=:red)
# # plt = plot!(Shape(cx4,cy4),color=:red,label="",linecolor=:red)
# # # plt = plot(Shape(cx5,cy5),color=:red,label="",linecolor=:red)
#
# for i = 1:N
#     x_sample_pos = [X_sample[i][t][1] for t = 1:T]
#     y_sample_pos = [X_sample[i][t][2] for t = 1:T]
#     plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,
#         color=:cyan,label= i != 1 ? "" : "sample")
# end
# display(plt)
#
# plt = scatter!(x_nom_pos,y_nom_pos,aspect_ratio=:equal,xlabel="x",ylabel="y",width=4.0,label="TO",color=:purple,legend=:topleft)
# x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
# y_sample_pos = [X_nom_sample[t][2] for t = 1:T]
# plt = plot!(x_sample_pos,y_sample_pos,aspect_ratio=:equal,width=4.0,label="DPO",color=:orange,legend=:bottomright)
#
# savefig(plt,joinpath(@__DIR__,"results/quadrotor_trajectory.png"))
#
# # Control
# plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',color=:purple,width=2.0,
#     title="car",xlabel="time (s)",ylabel="control",label=["v (nominal)" "ω (nominal)"],
#     legend=:bottom,linetype=:steppost)
# plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
#     width=2.0,label=["v (sample)" "ω (sample)"],linetype=:steppost)
# savefig(plt,joinpath(@__DIR__,"results/quadrotor_control.png"))
#
# # Samples
#
# # State samples
# plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
# for i = 1:N
#     t_sample = zeros(T)
#     for t = 2:T
#         t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
#     end
#     plt1 = plot!(t_sample,hcat(X_sample[i]...)',label="");
# end
# plt1 = plot!(t_sample,hcat(X_nom_sample...)',color=:red,width=2.0,
#     label=["nominal" "" ""])
# display(plt1)
# savefig(plt1,joinpath(@__DIR__,"results/quadrotor_sample_states.png"))
#
# # Control samples
# plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
# for i = 1:N
#     t_sample = zeros(T)
#     for t = 2:T
#         t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
#     end
#     plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
#         linetype=:steppost);
# end
# plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
#     label=["nominal" ""],linetype=:steppost)
# display(plt2)
# savefig(plt2,joinpath(@__DIR__,"results/quadrotor_sample_controls.png"))
#
# # using PGFPlots
# # const PGF = PGFPlots
# #
# # # TO trajectory
# # p_nom = PGF.Plots.Linear(hcat(X_nom...)[1,:],hcat(X_nom...)[2,:],
# #     mark="",style="color=purple, line width=3pt, solid",legendentry="TO")
# #
# # # DPO trajectory
# # p_dpo = PGF.Plots.Linear(hcat(X_nom_sample...)[1,:],hcat(X_nom_sample...)[2,:],
# #     mark="",style="color=orange, line width=3pt, solid",legendentry="DPO")
# #
# # # DPO trajectory
# # p_sample = [PGF.Plots.Linear(hcat(X_sample[i]...)[1,:],hcat(X_sample[i]...)[2,:],
# #     mark="",style="color=gray, line width=1pt, solid") for i = 1:N]
# # p_sample[6].legendentry="sample"
# #
# # # obstacles
# # p_circle = [PGF.Plots.Circle(circle..., style="color=black,fill=black") for circle in circles]
# #
# # a = Axis([p_circle;
# #     p_sample[1];
# #     p_sample[2];
# #     p_sample[3];
# #     p_sample[4];
# #     p_sample[5];
# #     p_sample[6];
# #     p_nom;
# #     p_dpo
# #     ],
# #     xmin=-0.4, ymin=-0.1, xmax=1.4, ymax=1.1,
# #     axisEqualImage=true,
# #     hideAxis=false,
# # 	ylabel="y",
# # 	xlabel="x",
# # 	legendStyle="{at={(0.01,0.99)},anchor=north west}",
# # 	)
# #
# # # Save to tikz format
# # dir = joinpath(@__DIR__,"results")
# # PGF.save(joinpath(dir,"car_obstacles.tikz"), a, include_preamble=false)
# #
# #
# # # visualize
# #
# # function visualize!(vis,p,q; Δt=0.1,r=0.25)
# #
# #     obj_path = joinpath(pwd(),"/home/taylor/Research/contact_implicit_trajectory_optimization/models/cybertruck/cybertruck.obj")
# #     mtl_path = joinpath(pwd(),"/home/taylor/Research/contact_implicit_trajectory_optimization/models/cybertruck/cybertruck.mtl")
# #
# #     ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.05)
# #     setobject!(vis["cybertruck"],ctm,MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
# #     settransform!(vis["cybertruck"], LinearMap(RotZ(pi)*RotX(pi/2.0)))
# #
# #     anim = MeshCat.Animation(convert(Int,floor(1/Δt)))
# #
# #     for t = 1:length(q)
# #
# #         MeshCat.atframe(anim,t) do
# #             x = [q[t][1];q[t][2];0.0]
# #             settransform!(vis["cybertruck"], compose(Translation(x),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
# #         end
# #     end
# #     # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
# #     MeshCat.setanimation!(vis,anim)
# # end
# #
# # function ModifiedMeshFileObject(obj_path::String, material_path::String; scale::T=0.1) where {T}
# #     obj = MeshFileObject(obj_path)
# #     rescaled_contents = rescale_contents(obj_path, scale=scale)
# #     material = select_material(material_path)
# #     mod_obj = MeshFileObject(
# #         rescaled_contents,
# #         obj.format,
# #         material,
# #         obj.resources,
# #         )
# #     return mod_obj
# # end
# #
# # function rescale_contents(obj_path::String; scale::T=0.1) where T
# #     lines = readlines(obj_path)
# #     rescaled_lines = copy(lines)
# #     for (k,line) in enumerate(lines)
# #         if length(line) >= 2
# #             if line[1] == 'v'
# #                 stringvec = split(line, " ")
# #                 vals = map(x->parse(Float64,x),stringvec[2:end])
# #                 rescaled_vals = vals .* scale
# #                 rescaled_lines[k] = join([stringvec[1]; string.(rescaled_vals)], " ")
# #             end
# #         end
# #     end
# #     rescaled_contents = join(rescaled_lines, "\r\n")
# #     return rescaled_contents
# # end
# #
# # function select_material(material_path::String)
# #     mtl_file = open(material_path)
# #     mtl = read(mtl_file, String)
# #     return mtl
# # end
# #
# # using Colors
# # using CoordinateTransformations
# # using FileIO
# # using GeometryTypes
# # using LinearAlgebra
# # using MeshCat
# # using MeshIO
# # using Rotations
# #
# # vis = Visualizer()
# # open(vis)
# # visualize!(vis,model,X_nom,Δt=H_nom_sample[1])
# #
# # for i = 1:4
# #     cyl = Cylinder(Point3f0(xc[i],yc[i],0),Point3f0(xc[i],yc[i],0.1),convert(Float32,0.035))
# #     setobject!(vis["cyl$i"],cyl,MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
# # end
#
#
#
#
# x1_sample
