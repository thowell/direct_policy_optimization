include("../src/sample_trajectory_optimization.jl")
include("../dynamics/dubins.jl")
using Plots

# Horizon
T = 5

# Bounds

# ul <= u <= uu
uu = 5.0
ul = -5.0

# h = h0 (fixed timestep)
tf0 = 1.0
h0 = tf0/(T-1)
hu = 0.05
hl = 0.0

# Initial and final states
x1 = [0.0; 0.0; 0.0]
xT = [1.0; 1.0; 0.0]

# Circle obstacle
r = 0.1
xc1 = 0.85
yc1 = 0.3
xc2 = 0.375
yc2 = 0.75
xc3 = 0.25
yc3 = 0.25
xc4 = 0.75
yc4 = 0.75

# Constraints
function con_obstacles!(c,x,u)
    c[1] = circle_obs(x[1],x[2],xc1,yc1,r)
    c[2] = circle_obs(x[1],x[2],xc2,yc2,r)
    c[3] = circle_obs(x[1],x[2],xc3,yc3,r)
    c[4] = circle_obs(x[1],x[2],xc4,yc4,r)
    nothing
end
m_con_obstacles = 4

# Objective
Q = [t < T ? Diagonal(rand(model.nx)) : Diagonal(rand(model.nx)) for t = 1:T]
R = [Diagonal(rand(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    integration=rk3_implicit,
                    goal_constraint=true,
                    con=con_obstacles!,
                    m_con=m_con_obstacles
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.01*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
# @time Z_nominal = solve(prob_moi,copy(Z0))

solve(prob_sample_moi,Z0_sample)


nx = model.nx
nu = model.nu
N = 2*nx
m_stage = prob.m_con

idx_nom = init_indices(nx,nu,T,time=true,shift=0)
idx_nom_z = vcat(idx_nom.x...,idx_nom.u...,idx_nom.h...)

shift = nx*T + nu*(T-1) + (T-1)
idx_sample = [init_indices(nx,nu,T,time=false,shift=shift + (i-1)*(nx*T + nu*(T-1))) for i = 1:N]
shift += N*(nx*T + nu*(T-1))
idx_x_tmp = [init_indices(nx,0,T-1,time=false,shift=shift + (i-1)*(nx*(T-1))) for i = 1:N]
shift += N*(nx*(T-1))
idx_K = [shift + (t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]

n_nlp = (nx*T + nu*(T-1) + (T-1)) + N*(nx*T + nu*(T-1)) + N*(nx*(T-1)) + nu*nx*(T-1)
m_sample_nlp = N*2*nx*(T-2) + N*nu*(T-1) + N*m_stage*(T-2)
idx_nom.x[1]
idx_sample[1].x[1]
function obj_sample(z,idx_nom,idx_sample,Q,R,T,N)
    J = 0.0

    # sample
    for t = 1:T-1
        u_nom = view(z,idx_nom.u[t])
        x⁺_nom = view(z,idx_nom.x[t+1])

        for i = 1:N
            ui = view(z,idx_sample[i].u[t])
            xi⁺ = view(z,idx_sample[i].x[t+1])
            println((xi⁺ - x⁺_nom)'*Q[t+1]*(xi⁺ - x⁺_nom))
            J += (xi⁺ - x⁺_nom)'*Q[t+1]*(xi⁺ - x⁺_nom) + (ui - u_nom)'*R[t]*(ui - u_nom)
        end
    end

    return J
end

z0 = rand(n_nlp)
z0
obj_sample(z0,idx_nom,idx_sample,Q,R,T,N)

function con_sample!(c,z,idx_nom,idx_sample,idx_K,Q,R,models,β,w,con,m_con,T,N,integration)
    shift = 0

    # dynamics + resampling (x1 is taken care of w/ primal bounds)
    β = 1.0
    w = 1.0e-1
    for t = 2:T-1
        h = view(z,idx_nom.h[t])
        x⁺_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        xs⁺ = resample(x⁺_tmp,β=β,w=w) # resample

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            xi⁺ = view(z,idx_sample[i].x[t+1])

            c[shift .+ (1:nx)] = integration(models[i],x⁺_tmp[i],xi,ui,h)
            shift += nx
            c[shift .+ (1:nx)] = xs⁺[i] - xi⁺
            shift += nx
        end
    end

    # controller for samples
    for t = 1:T-1
        x_nom = view(z,idx_nom.x[t])
        u_nom = view(z,idx_nom.u[t])
        K = reshape(view(z,idx_K[t]),nu,nx)

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            c[shift .+ (1:nu)] = ui + K*(xi - x_nom) - u_nom
            shift += nu
        end
    end

    # stage constraints samples
    for t = 2:T-1
        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])

            con(view(c,shift .+ (1:m_con)),xi,ui)
            shift += m_con
        end
    end

    nothing
end

z0 = rand(n_nlp)
obj_sample(z0)
c0 = zeros(m_sample_nlp)
con_sample!(c0,z0,idx_nom,idx_sample,idx_K,Q,R,[model for i = 1:2:nx],1.0,1.0e-1,prob.con,prob.m_con,T,N,prob.integration)
