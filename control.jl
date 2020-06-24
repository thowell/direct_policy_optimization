function tvlqr(z_nom,u_nom,model,Δt)
    A = []
    B = []
    for t = 1:T-1
        fz(z) = midpoint(model,z,u_nom[t],Δt)
        fu(u) = midpoint(model,z_nom[t],u,Δt)
        push!(A,ForwardDiff.jacobian(fz,z_nom[t]))
        push!(B,ForwardDiff.jacobian(fu,u_nom[t]))
    end

    P = []
    push!(P,Qf)
    K = []

    for t = T-1:-1:1
        push!(K,(R + B[t]'*P[end]*B[t])\(B[t]'*P[end]*A[t]))
        push!(P,Q + A[t]'*P[end]*A[t] - (A[t]'*P[end]*B[t])*K[end])
    end

    return K, P , A, B
end

function simulate_linear_controller(K,z_nom,u_nom,T_sim,Δt,z0,w)
    T = length(K)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0 + w[1]]
    u_rollout = []
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end] + w[k+1]
        u = u_nom[k] - K[k]*(z - z_nom[k])
        push!(z_rollout,midpoint(model,z,u,dt_sim))
        push!(u_rollout,u)
    end
    return z_rollout, u_rollout
end
