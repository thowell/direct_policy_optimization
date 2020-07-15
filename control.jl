function simulate_linear_controller(Kc,z_nom,u_nom,model,Q,R,T_sim,Δt,z0,w;
        _norm=2,
        ul=-Inf*ones(length(u_nom[1])),
        uu=Inf*ones(length(u_nom[1])))
    T = length(Kc)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0]
    u_rollout = []
    J = 0.0
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end] + dt_sim*w[:,tt]
        u = u_nom[k] - Kc[k]*(z - z_nom[k])

        # clip controls
        u = max.(u,ul)
        u = min.(u,uu)

        push!(z_rollout,dynamics(model,z,u,dt_sim))
        push!(u_rollout,u)
        if _norm == 2
            J += (z_rollout[end]-z_nom[k+1])'*Q[k+1]*(z_rollout[end]-z_nom[k+1])
            J += (u_rollout[end]-u_nom[k])'*R[k]*(u_rollout[end]-u_nom[k])
        else
            J += norm(sqrt(Q[k+1])*(z_rollout[end]-z_nom[k+1]),_norm) + norm(sqrt(R[k])*(u-u_nom[k]),_norm)
        end
    end
    return z_rollout, u_rollout, J/(T_sim-1)
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
