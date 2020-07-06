function simulate_linear_controller(Kc,z_nom,u_nom,T_sim,Δt,z0,w)
    T = length(K)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0 + w[:,1]]
    u_rollout = []
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end] + w[:,1]
        u = u_nom[k] - Kc[k]*(z - z_nom[k])
        push!(z_rollout,dynamics(z,u,dt_sim))
        push!(u_rollout,u)
    end
    return z_rollout, u_rollout
end
