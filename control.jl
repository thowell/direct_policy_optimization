function simulate_linear_controller(Kc,z_nom,u_nom,Q,R,T_sim,Δt,z0,w)
    T = length(Kc)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0 + w[:,1]]
    u_rollout = []
    J = 0.0
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end] + w[:,tt]
        u = u_nom[k] - Kc[k]*(z - z_nom[k])

        push!(z_rollout,dynamics(z,u,dt_sim))
        push!(u_rollout,u)
        J += (z-z_nom[k])'*Q[k]*(z-z_nom[k]) + (u-u_nom[k])'*R[k]*(u-u_nom[k])
    end
    return z_rollout, u_rollout, J
end

function nominal_trajectories(z_nom,u_nom,T_sim,Δt)
    T = length(z_nom)
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    _z_nom = [z_nom[1]]
    _u_nom = []
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)

        push!(_z_nom,z_nom[k])
        push!(_u_nom,u_nom[k])
    end
    return _z_nom, _u_nom
end
