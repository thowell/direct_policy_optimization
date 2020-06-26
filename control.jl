function tvlqr(z_nom,u_nom,model,Q,R,Qf,Δt)
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
#
# function unscented_tvlqr(z_nom,u_nom,model,Q,R,Qf,n,m,Δt;β=1.0,N_sample=2*(n+m))
#     K_u = []
#
#     tmp = zeros(n+m,n+m)
#     z_tmp = zeros(n+m)
#     H = Qf
#
#     for t = T:-1:2
#         # println("t: $t")
#         tmp[1:n,1:n] = H
#         tmp[n .+ (1:m),n .+ (1:m)] = R
#         L = Array(cholesky(Hermitian(inv(tmp))))
#
#         z_tmp[1:n] = z_nom[t]
#         z_tmp[n .+ (1:m)] = u_nom[t-1]
#
#         z_sample = [z_tmp + β*L[:,i] for i = 1:(n+m)]
#         z_sample = [z_sample...,[z_tmp - β*L[:,i] for i = 1:(n+m)]...]
#
#         z_sample_prev = [[midpoint(model,zs[1:n],zs[n .+ (1:m)],-Δt);zs[n .+ (1:m)]] for zs in z_sample]
#
#         M = 0.5/(β^2)*sum([(zsp - z_tmp)*(zsp - z_tmp)' for zsp in z_sample_prev])
#
#         P = inv(M)
#         P[1:n,1:n] += Q
#
#         A = P[1:n,1:n]
#         C = P[n .+ (1:m),1:n]
#         B = P[n .+ (1:m), n .+ (1:m)]
#
#         K = B\C
#
#         push!(K_u,K)
#
#         H = A + K'*B*K - K'*C - C'*K
#     end
#     return K_u
# end



function simulate_linear_controller(K,z_nom,u_nom,T_sim,Δt,z0,w,integration)
    T = length(K)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0 + vec(rand(w,1))]
    u_rollout = []
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end] + vec(rand(w,1))
        u = u_nom[k] - K[k]*(z - z_nom[k])
        push!(z_rollout,integration(model,z,u,dt_sim))
        push!(u_rollout,u)
    end
    return z_rollout, u_rollout
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
