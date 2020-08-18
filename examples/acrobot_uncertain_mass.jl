include("../src/sample_trajectory_optimization.jl")
include("../dynamics/acrobot.jl")
using Plots


# Horizon
T = 51

# Bounds

# ul <= u <= uu
uu = 5.0
ul = -5.0

# hl <= h <= hu
tf0 = 5.0
h0 = tf0/(T-1) # timestep

hu = h0
hl = h0

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]

# Objective
Q = [t<T ? Diagonal(ones(model.nx)) : Diagonal(10.0*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-2*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal(10.0*ones(model.nx)) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Models
model1 = Acrobot(1.0,1.0,1.0,0.5,1.35,1.0,1.0,0.5,9.81,nx,nu)
model2 = Acrobot(1.0,1.0,1.0,0.5,1.3,1.0,1.0,0.5,9.81,nx,nu)
model3 = Acrobot(1.0,1.0,1.0,0.5,1.2,1.0,1.0,0.5,9.81,nx,nu)
model4 = Acrobot(1.0,1.0,1.0,0.5,1.1,1.0,1.0,0.5,9.81,nx,nu)
model5 = Acrobot(1.0,1.0,1.0,0.5,0.9,1.0,1.0,0.5,9.81,nx,nu)
model6 = Acrobot(1.0,1.0,1.0,0.5,0.8,1.0,1.0,0.5,9.81,nx,nu)
model7 = Acrobot(1.0,1.0,1.0,0.5,0.7,1.0,1.0,0.5,9.81,nx,nu)
model8 = Acrobot(1.0,1.0,1.0,0.5,0.65,1.0,1.0,0.5,9.81,nx,nu)

# Problem
prob_nom = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)

prob1 = init_problem(model.nx,model.nu,T,x1,xT,model1,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob2 = init_problem(model.nx,model.nu,T,x1,xT,model2,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob3 = init_problem(model.nx,model.nu,T,x1,xT,model3,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob4 = init_problem(model.nx,model.nu,T,x1,xT,model4,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob5 = init_problem(model.nx,model.nu,T,x1,xT,model5,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob6 = init_problem(model.nx,model.nu,T,x1,xT,model6,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob7 = init_problem(model.nx,model.nu,T,x1,xT,model7,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)
prob8 = init_problem(model.nx,model.nu,T,x1,xT,model8,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)

# MathOptInterface problem
prob_nom_moi = init_MOI_Problem(prob_nom)

prob1_moi = init_MOI_Problem(prob1)
prob2_moi = init_MOI_Problem(prob2)
prob3_moi = init_MOI_Problem(prob3)
prob4_moi = init_MOI_Problem(prob4)
prob5_moi = init_MOI_Problem(prob5)
prob6_moi = init_MOI_Problem(prob6)
prob7_moi = init_MOI_Problem(prob7)
prob8_moi = init_MOI_Problem(prob8)

# Initialization
X0 = linear_interp(x1,xT,T) # linear interpolation for states
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_nom_moi,copy(Z0),nlp=:SNOPT7)

@time Z_nominal1 = solve(prob1_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal2 = solve(prob2_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal3 = solve(prob3_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal4 = solve(prob4_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal5 = solve(prob5_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal6 = solve(prob6_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal7 = solve(prob7_moi,copy(Z0),nlp=:SNOPT7)
@time Z_nominal8 = solve(prob8_moi,copy(Z0),nlp=:SNOPT7)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob1)

X_nominal1, U_nominal1, H_nominal1 = unpack(Z_nominal1,prob1)
X_nominal2, U_nominal2, H_nominal2 = unpack(Z_nominal2,prob2)
X_nominal3, U_nominal3, H_nominal3 = unpack(Z_nominal3,prob3)
X_nominal4, U_nominal4, H_nominal4 = unpack(Z_nominal4,prob4)
X_nominal5, U_nominal5, H_nominal5 = unpack(Z_nominal5,prob5)
X_nominal6, U_nominal6, H_nominal6 = unpack(Z_nominal6,prob6)
X_nominal7, U_nominal7, H_nominal7 = unpack(Z_nominal7,prob7)
X_nominal8, U_nominal8, H_nominal8 = unpack(Z_nominal8,prob8)

plot(hcat(X_nominal1...)',xlabel="time step",label="")
plot!(hcat(X_nominal2...)',xlabel="time step",label="")
plot!(hcat(X_nominal3...)',xlabel="time step",label="")
plot!(hcat(X_nominal4...)',xlabel="time step",label="")
plot!(hcat(X_nominal5...)',xlabel="time step",label="")
plot!(hcat(X_nominal6...)',xlabel="time step",label="")
plot!(hcat(X_nominal7...)',xlabel="time step",label="")
plot!(hcat(X_nominal8...)',xlabel="time step",label="")
plot!(hcat(X_nominal...)',xlabel="time step",width=2.0,color=:red,label="")

plot(hcat(U_nominal1...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal2...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal3...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal4...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal5...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal6...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal7...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal8...)',xlabel="time step",linetype=:steppost,label="")
plot!(hcat(U_nominal...)',xlabel="time step",linetype=:steppost,width=2.0,color=:red,label="")

vis = Visualizer()
open(vis)
visualize!(vis,model,[X_nominal,X_nominal1,X_nominal2,X_nominal3,X_nominal4,X_nominal5,X_nominal6,X_nominal7,X_nominal8],
    color=[RGBA(1,0,0,0.5),RGBA(0,0,0,1.0),RGBA(1,0,0,1.0),RGBA(0,1,0,1.0),RGBA(0,0,1,1.0),RGBA(1,1,0,1.0),RGBA(1,0,1,1.0),RGBA(0,1,1,1.0),RGBA(1,1,1,1.0)],Δt=h0)

Xs_nominal = [
              X_nominal1,
              X_nominal2,
              X_nominal3,
              X_nominal4,
              X_nominal5,
              X_nominal6,
              X_nominal7,
              X_nominal8
              ]
Us_nominal = [
            U_nominal1,
            U_nominal2,
            U_nominal3,
            U_nominal4,
            U_nominal5,
            U_nominal6,
            U_nominal7,
            U_nominal8
            ]

Hs_nominal = [
            H_nominal1,
            H_nominal2,
            H_nominal3,
            H_nominal4,
            H_nominal5,
            H_nominal6,
            H_nominal7,
            H_nominal8
            ]
# Samples
N = 2*model.nx
models = [model1,model2,model3,model4,model5,model6,model7,model8]
β = 1.0
w = 1.0e-8*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)
K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ)
prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = zeros(prob_sample.N_nlp)

for t = 1:T
    Z0_sample[prob_sample.idx_nom.x[t]] = X_nominal[t]

    t>T-1 && continue
    Z0_sample[prob_sample.idx_nom.u[t]] = U_nominal[t]
    Z0_sample[prob_sample.idx_nom.h[t]] = H_nominal[t]
end

for t = 1:T
    for i = 1:N
        Z0_sample[prob_sample.idx_sample[i].x[t]] = Xs_nominal[i][t]

        t>T-1 && continue
        Z0_sample[prob_sample.idx_sample[i].u[t]] = Us_nominal[i][t]
        Z0_sample[prob_sample.idx_sample[i].h[t]] = Hs_nominal[i][t]
    end
end

for t = 1:T-1
    Z0_sample[prob_sample.idx_K[t]] = vec(K[t])
end

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,nlp=:ipopt)
# Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,nlp=:ipopt,time_limit=600)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

# Plot results

# Time
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nominal))s")
display("time (sample nominal): $(sum(H_nom_sample))s")

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nominal...))',
    color=:purple,width=2.0,title="Acrobot",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/acrobot_control_mass.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ1 (nominal)",title="Acrobot",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="θ2 (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ1 (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="θ2 (sample)")
savefig(plt,joinpath(@__DIR__,"results/acrobot_state_mass.png"))

# State samples
plt1 = plot(t_sample,hcat(X_nom_sample...)[1,:],color=:red,width=2.0,title="",
    label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1,:],label="");
end

plt2 = plot(t_sample,hcat(X_nom_sample...)[2,:],color=:red,width=2.0,label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt2 = plot!(t_sample,hcat(X_sample[i]...)[2,:],label="");
end
plt12 = plot(plt1,plt2,layout=(2,1),title=["θ1" "θ2"],xlabel="time (s)")
savefig(plt,joinpath(@__DIR__,"results/acrobot_sample_state.png"))

# Control samples
plt3 = plot(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=2.0,
    title="Control",label="",xlabel="time (s)");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1,:],label="");
end
display(plt3)
savefig(plt,joinpath(@__DIR__,"results/acrobot_sample_control.png"))
