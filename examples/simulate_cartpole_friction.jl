include("../tests/ipopt.jl")

function simulate_cartpole_friction(Kc,z_nom,u_nom,model,Q,R,T_sim,Δt,z0,w;
        _norm=2,
        ul=-Inf*ones(length(u_nom[1])),
        uu=Inf*ones(length(u_nom[1])),
        friction=false,
        μ=0.1)

    T = length(Kc)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    t_ctrl = range(0,stop=tf,length=T)
    dt_sim = tf/(T_sim-1)

    u_policy = 1:model.nu_policy

    A_state = hcat(z_nom...)
    A_ctrl = hcat(u_nom...)

    z_rollout = [z0]
    u_rollout = []
    J = 0.0
    Jx = 0.0
    Ju = 0.0
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)

        z_cubic = zeros(model.nx)
        for i = 1:model.nx
            interp_cubic = CubicSplineInterpolation(t_ctrl, A_state[i,:])
            z_cubic[i] = interp_cubic(t)
        end

        z = z_rollout[end] + dt_sim*w[:,tt]
        u = u_nom[k][1:model.nu_policy] - Kc[k]*(z - z_cubic)

        # clip controls
        u = max.(u,ul[u_policy])
        u = min.(u,uu[u_policy])

        if friction
            _u = [u[1]-μ*sign(z_cubic[3])*model.g*(model.mp+model.mc);0.0;0.0]
        else
            _u = u[1]
        end

        push!(z_rollout,rk3(model,z,_u,dt_sim))
        push!(u_rollout,u)

        if _norm == 2
            J += (z_rollout[end]-z_cubic)'*Q[k+1]*(z_rollout[end]-z_cubic)
            J += (u_rollout[end]-u_nom[k][u_policy])'*R[k][u_policy,u_policy]*(u_rollout[end]-u_nom[k][u_policy])
            Jx += (z_rollout[end]-z_cubic)'*Q[k+1]*(z_rollout[end]-z_cubic)
            Ju += (u_rollout[end]-u_nom[k][u_policy])'*R[k][u_policy,u_policy]*(u_rollout[end]-u_nom[k][u_policy])
        else
            J += norm(sqrt(Q[k+1])*(z_rollout[end]-z_cubic),_norm)
            J += norm(sqrt(R[k][u_policy,u_policy])*(u-u_nom[k][u_policy]),_norm)
            Jx += norm(sqrt(Q[k+1])*(z_rollout[end]-z_cubic),_norm)
            Ju += norm(sqrt(R[k][u_policy,u_policy])*(u-u_nom[k][u_policy]),_norm)
        end
    end
    return z_rollout, u_rollout, J/(T_sim-1), Jx/(T_sim-1), Ju/(T_sim-1)
end

using Distributions
T_sim = 10T
Δt = h0

W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w0 = rand(W0,1)


model_sim = model_friction
model_sim.μ = 0.1

t_sim_nominal = range(0,stop=H_nominal[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)


K_nominal = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr,u_policy=(1:1))

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_cartpole_friction(K_nominal,X_nominal,U_nominal,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nominal[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    )
plt_tvlqr_nom = plot(t_nominal,hcat(X_nominal...)[1:2,:]',legend=:topleft,color=:red,label=["nominal (no friction)" ""],
    width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state",ylims=(-1,5))
plt_tvlqr_nom = plot!(t_sim_nominal,hcat(z_tvlqr...)[1:2,:]',color=:purple,label=["tvlqr" ""],width=2.0)
savefig(plt_tvlqr_nom,joinpath(@__DIR__,"results/cartpole_friction_tvlqr_nom_sim.png"))

K_friction_nominal = TVLQR_gains(model,X_friction_nominal,U_friction_nominal,H_friction_nominal,Q_lqr,R_lqr,u_policy=(1:1))

z_tvlqr_friction, u_tvlqr_friction, J_tvlqr_friction, Jx_tvlqr_friction, Ju_tvlqr_friction = simulate_cartpole_friction(K_friction_nominal,X_friction_nominal,U_friction_nominal,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_friction_nominal[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    )
plt_tvlqr_friction = plot(t_nominal,hcat(X_friction_nominal...)[1:2,:]',color=:red,label=["nominal (sample)" ""],
    width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state",legend=:topleft)
plt_tvlqr_friction = plot!(t_sim_nominal,hcat(z_tvlqr_friction...)[1:2,:]',color=:magenta,label=["tvlqr" ""],width=2.0)
savefig(plt_tvlqr_friction,joinpath(@__DIR__,"results/cartpole_friction_tvlqr_friction_sim.png"))

z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_cartpole_friction(K_sample,X_nom_sample,U_nom_sample,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nom_sample[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    )

plt_sample = plot(t_nominal,hcat(X_friction_nominal...)[1:2,:]',legend=:bottom,color=:red,label=["nominal (sample)" ""],
    width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state")
plt_sample = plot!(t_sim_nominal,hcat(z_sample...)[1:2,:]',color=:orange,
    label=["sample" ""],width=2.0,legend=:topleft)
savefig(plt_sample,joinpath(@__DIR__,"results/cartpole_friction_sample_sim.png"))

# objective value
J_tvlqr
J_tvlqr_friction
J_sample

# state tracking
Jx_tvlqr
Jx_tvlqr_friction
Jx_sample

# control tracking
Ju_tvlqr
Ju_tvlqr_friction
Ju_sample
