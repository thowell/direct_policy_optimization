include("../src/sample_trajectory_optimization.jl")
include("../dynamics/hopper3D.jl")
using Plots

# Horizon
T = 10
# Tm = convert(Int,(T-3)/2 + 3)

tf = 1.0
Δt = tf/T

# Initial and final states
x1 = [0.0, 0., 0.5*model.r, 0., 0., 0., 0.5*model.r]
xT = [(0.5*model.r + 0.25 -0.5*model.r+0.2)/2, 0.25, 0.5*model.r + 0.25, 0.0, 0, 0., 0.5*model.r]
x_ref = [x1,linear_interp(x1,xT,T-1)...]
visualize!(vis,model,x_ref,Δt=Δt)

ϕ_func(model,x1)
ϕ_func(model,xT)

# Bounds
# xl <= x <= xu
xu_traj = [copy(model.qU) for t=1:T]
xl_traj = [copy(model.qL) for t=1:T]

# xu_traj = [x1 for t=1:T]
# xl_traj = [x1 for t=1:T]

xu_traj[1] = copy(x1)
xl_traj[1] = copy(x1)

xu_traj[2] = copy(x1)
xl_traj[2] = copy(x1)

xu_traj[T] = copy(xT)
xl_traj[T] = copy(xT)
# xu_traj[11][1:3] = copy(xM[1:3])
# xl_traj[11][1:3] = copy(xM[1:3])
#
# xu_traj[21][1:3] = copy(xN[1:3])
# xl_traj[21][1:3] = copy(xN[1:3])
#
# xu_traj[31] = copy(xT)
# xl_traj[31] = copy(xT)

# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

ul_traj = [ul for t = 1:T-2]
uu_traj = [uu for t = 1:T-2]

# h = h0 (fixed timestep)
hu = Δt
hl = Δt

# Objective
Q = [(t<T ? Diagonal(1.0*[1.0,1.0,1.0,1.0e-1,1.0e-1,1.0e-1,1.0])
        : Diagonal(100.0*[1.0,1.0,1.0,1.0e-1,1.0e-1,1.0e-1,1.0]) ) for t = 1:T]
R = [Diagonal([1.0,1.0,1.0e-2]) for t = 1:T-2]
c = 0.0

x_ref = [x1,linear_interp(x1,xT,T-1)...]
for t = 3:T-1
    x_ref[t][7] = 0.1*model.r
end
u_ref = [0.0;0.0;Δt*model.g*(model.mb + model.ml)]
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[u_ref for t=1:T-2])
model.α = 100.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2]
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = deepcopy(x_ref)
U0 = [[u_ref;1.0e-5*rand(model.nu-model.nu_ctrl)] for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:ipopt)
@time Z_nominal = solve(prob_moi,copy(Z_nominal),nlp=:ipopt)

X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

# x_nom = [X_nom[t][1] for t = 1:T]
# z_nom = [X_nom[t][2] for t = 1:T]
# λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
# b_nom = [U_nom[t][model.idx_b] for t = 1:T-2]
# ψ_nom = [U_nom[t][model.idx_ψ[1]] for t = 1:T-2]
# η_nom = [U_nom[t][model.idx_η] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)
#
# plot(hcat(x_ref...)[1:1,:]')
# plot!(x_nom)
# plot(hcat(x_ref...)[2:2,:]')
# plot!(z_nom)
#
# plot(λ_nom,linetype=:steppost)
# plot(hcat(b_nom...)',linetype=:steppost)
# plot(ψ_nom,linetype=:steppost)
# plot(hcat(η_nom...)',linetype=:steppost)
# plot(hcat(U_nom...)',linetype=:steppost)

using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshIO
using Rotations
using Meshing

vis = Visualizer()
open(vis)
visualize!(vis,model,X_nom,Δt=H_nom[1])
U_nom[1][1:3]
# plot(hcat(U_nom...)[1:3,:]')

X_nom[11][1:3] - xM[1:3]
ϕ_func(model,X_nom[T])

norm(X_nom[Tm] - xM)

ϕ_func(model,X_nom[T])
X_nom[T]
m1w = 2
m2w = -2
bw = 0.2
f1 = x -> x[3] - (m1w*x[1] - bw)
f2 = x -> x[3] - (m2w*x[1] - bw)

sdf1 = SignedDistanceField(f1, HyperRectangle(Vec(-1, 0, 0), Vec(10, 2, 1)))
mesh1 = HomogenousMesh(sdf1, MarchingTetrahedra())
setobject!(vis["slope1"], mesh1,
           MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))

sdf2 = SignedDistanceField(f2, HyperRectangle(Vec(-1, 0, 0), Vec(10, 2, 1)))
mesh2 = HomogenousMesh(sdf2, MarchingTetrahedra())
setobject!(vis["slope2"], mesh2,
      MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))
