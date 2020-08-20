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


# μ = 0.1
model_sim = model_friction
model_sim.μ = 0.1

t_sim_nominal = range(0,stop=H_nominal[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)

z_sample1, u_sample1, J_sample1, Jx_sample1, Ju_sample1 = simulate_cartpole_friction(K_sample,X_nom_sample,U_nom_sample,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nom_sample[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    )

z_sample_c1, u_sample_c1, J_sample_c1, Jx_sample_c1, Ju_sample_c1 = simulate_cartpole_friction(K_sample_coefficients,X_nom_sample_coefficients,U_nom_sample_coefficients,
    model_sim,Q_lqr,R_lqr,T_sim,Δt,X_nom_sample_coefficients[1],w,ul=ul_friction,uu=uu_friction,friction=true,
    )

plot(hcat(z_sample1...)')
plot!(hcat(z_sample_c1...)')

plot(hcat(u_sample1...)')
plot!(hcat(u_sample_c1...)')

# objective value
J_sample1
J_sample_c1

# state tracking
Jx_sample1
Jx_sample_c1

# control tracking
Ju_sample1
Ju_sample_c1
