include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/quadrotor.jl"))
include(joinpath(pwd(),"dynamics/obstacles.jl"))
# include(joinpath(pwd(),"dynamics/visualize.jl"))
println("quadrotor test loaded")

# Horizon
T = 51
Tm = convert(Int,floor(T/2)+1)

# Bounds

# ul <= u <= uu
uu = 5.0*ones(model.nu)
ul = zeros(model.nu)

uu_traj = [copy(uu) for t = 1:T-1]
ul_traj = [copy(ul) for t = 1:T-1]

# Circle obstacle
r_cyl = 0.5
r = r_cyl + model.L
xc1 = 2.0-0.125
yc1 = 2.0
xc2 = 4.0-0.125
yc2 = 4.0
# xc3 = 2.25
# yc3 = 1.0
# xc4 = 2.0
# yc4 = 4.0

xc = [xc1,xc2]#,xc3,xc4]
yc = [yc1,yc2]#,yc3,yc4]

circles = [(xc1,yc1,r),(xc2,yc2,r)]#,(xc3,yc3,r),(xc4,yc4,r)]

# Constraints
function c_stage!(c,x,u,t,model)
    c[1] = circle_obs(x[1],x[2],xc1,yc1,r)
    c[2] = circle_obs(x[1],x[2],xc2,yc2,r)
    # c[3] = circle_obs(x[1],x[2],xc3,yc3,r)
    # c[4] = circle_obs(x[1],x[2],xc4,yc4,r)
    nothing
end
m_stage = 2

# h = h0 (fixed timestep)
tf0 = 5.0
h0 = tf0/(T-1)
hu = h0
hl = 0.0*h0

# Initial and final states
x1 = zeros(model.nx)
x1[3] = 1.0
xT = copy(x1)
xT[1] = 5.0
xT[2] = 5.0

xl = -Inf*ones(model.nx)
xl[1] = -1.0
xl[2] = -1.0
xl[3] = 0.0

xu = Inf*ones(model.nx)
xu[1] = 6.0
xu[2] = 6.0
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
c = 10.0
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
                    stage_constraints=true,
                    m_stage=[m_stage for t=1:T-1]
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [copy(u_ref) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

println("problem setup")
#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
sum(H_nom)

println("*complete*")
