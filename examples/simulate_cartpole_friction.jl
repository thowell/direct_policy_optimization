function simulate_linear_controller_friction(Kc,z_nom,u_nom,model,Q,R,T_sim,Δt,z0,w;
        _norm=2,
        ul=-Inf*ones(length(u_nom[1])),
        uu=Inf*ones(length(u_nom[1])),
        pfl=false,
        friction=false,
        μ=0.5)
    T = length(Kc)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0]
    u_rollout = []
    J = 0.0
    Jx = 0.0
    Ju = 0.0
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end] + dt_sim*w[:,tt]
        u = u_nom[k] - Kc[k]*(z - z_nom[k])

        # clip controls
        u = max.(u,ul)
        u = min.(u,uu)

        if friction
            _u = u[1] + μ*sign(z_rollout[end][3])*model.g*(model.mp+model.mc)
        else
            _u = u[1]
        end

        push!(z_rollout,rk3(model,z,_u,dt_sim))
        push!(u_rollout,u)
        # if _norm == 2
        #     J += (z_rollout[end]-z_nom[k+1])'*Q[k+1]*(z_rollout[end]-z_nom[k+1])
        #     J += (u_rollout[end]-u_nom[k])'*R[k]*(u_rollout[end]-u_nom[k])
        #     Jx += (z_rollout[end]-z_nom[k+1])'*Q[k+1]*(z_rollout[end]-z_nom[k+1])
        #     Ju += (u_rollout[end]-u_nom[k])'*R[k]*(u_rollout[end]-u_nom[k])
        # else
        #     J += norm(sqrt(Q[k+1])*(z_rollout[end]-z_nom[k+1]),_norm)
        #     J += norm(sqrt(R[k])*(u-u_nom[k]),_norm)
        #     Jx += norm(sqrt(Q[k+1])*(z_rollout[end]-z_nom[k+1]),_norm)
        #     Ju += norm(sqrt(R[k])*(u-u_nom[k]),_norm)
        # end
    end
    return z_rollout, u_rollout, J/(T_sim-1), Jx/(T_sim-1), Ju/(T_sim-1)
end

R_nom = [R_lqr[t][1:model.nu] for t = 1:T-1]
K_nom = TVLQR_gains(model,X_friction_nominal,U_friction_nominal,H_friction_nominal,Q_lqr,R_lqr;
        u_policy=(1:model_nominal.nu_policy))

using Distributions
T_sim = 1T
W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-32*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-32*ones(nx)))
w0 = rand(W0,1)

friction = true
μ = 0.2
z0_sim = vec(copy(X_nominal[1]) + 1.0*w0[:,1])

# z_nom_sim, u_nom_sim = nominal_trajectories(X_nominal,U_nominal,T_sim,h0)

t_nom = range(0,stop=h0*(T-1),length=T)
t_sim = range(0,stop=h0*(T-1),length=T_sim)

plt1 = plot(t_nom,hcat(X_nominal...)[1,:],
    title="States",legend=:topleft,color=:red,label="nominal",width=2.0,xlabel="time (s)")
plt1 = plot!(t_nom,hcat(X_nominal...)[2,:],
    color=:red,label="",width=2.0)

z_tvlqr_bnds, u_tvlqr_bnds, J_tvlqr_bnds, Jx_tvlqr_bnds, Ju_tvlqr_bnds = simulate_linear_controller_friction(K_nom,
    X_friction_nominal,[U_friction_nominal[t][1:model.nu] for t = 1:T-1],model,Q_lqr,R_nom,
    T_sim,h0,z0_sim,w,friction=friction,μ=μ,
    ul=ul_friction[1:model.nu],uu=uu_friction[1:model.nu])

plt1 = plot!(t_sim,hcat(z_tvlqr_bnds...)[1,:],color=:blue,label="tvlqr (bnds)",width=2.0)
plt1 = plot!(t_sim,hcat(z_tvlqr_bnds...)[2,:],color=:blue,label="",width=2.0)

plt2 = plot(t_nom,hcat(X_robust...)[1,:],title="States",legend=:topleft,color=:red,label="nominal (robust)",width=2.0,xlabel="time (s)")
plt2 = plot!(t_nom,hcat(X_robust...)[2,:],color=:red,label="",width=2.0)

z_robust, u_robust, J_robust, Jx_robust, Ju_robust = simulate_linear_controller_friction(K_robust,
    X_nom_sample,[U_nom_sample[t][1:model.nu] for t = 1:T-1],model,Q_lqr,R_nom,
    T_sim,h0,z0_sim,w,friction=friction,μ=μ,ul=ul_friction,uu=uu_friction)
plt2 = plot!(t_sim,hcat(z_robust...)[1,:],color=:orange,label="robust (bnds)",width=2.0)
plt2 = plot!(t_sim,hcat(z_robust...)[2,:],color=:orange,label="",width=2.0)

plt3 = plot(t_nom,hcat(X_pfl_robust...)[1,:],title="States",legend=:topleft,color=:red,label="nominal (robust pfl)",width=2.0,xlabel="time (s)")
plt3 = plot!(t_nom,hcat(X_pfl_robust...)[2,:],color=:red,label="",width=2.0)

z_pfl_robust, u_pfl_robust, J_pfl_robust, Jx_pfl_robust, Ju_pfl_robust = simulate_linear_controller_friction(K_robust_pfl,X_pfl_robust,U_pfl_robust,model_pfl,Q_lqr,R_lqr,T_sim,h0,z0_sim,w,friction=friction,μ=μ,ul=ul*ones(nu),uu=uu*ones(nu),pfl=true)
plt3 = plot!(t_sim,hcat(z_pfl_robust...)[1,:],color=:cyan,label="robust pfl (bnds)",width=2.0)
plt3 = plot!(t_sim,hcat(z_pfl_robust...)[2,:],color=:cyan,label="",width=2.0)

plot(plt1,plt2,plt3,layout=(3,1))

plt4 = plot(t_nom[1:end-1],hcat(U_nominal...)[:],title="Controls",xlabel="time (s)",legend=:bottomright,color=:red,label="nominal",linetype=:steppost)
plt4 = plot!(t_sim[1:end-1],hcat(u_tvlqr_bnds...)[:],color=:blue,label="tvlqr (bnds)",linetype=:steppost)

plt5 = plot(t_nom[1:end-1],hcat(U_robust...)[:],xlabel="time (s)",legend=:bottomright,color=:red,label="robust nominal",linetype=:steppost)
plt5 = plot!(t_sim[1:end-1],hcat(u_robust...)[:],color=:orange,label="robust",linetype=:steppost)

plt6 = plot(t_nom[1:end-1],hcat(U_pfl_robust...)[:],xlabel="time (s)",legend=:bottomright,color=:red,label="robust pfl nominal",linetype=:steppost)
u_pfl_robust_conv = [u_pfl_robust[t][1]*(model.mc + model.mp*sin(z_pfl_robust[t][2])*sin(z_pfl_robust[t][2])) - model.mp*sin(z_pfl_robust[t][2])*(model.l*z_pfl_robust[t][4]*z_pfl_robust[t][4] + model.g*cos(z_pfl_robust[t][2])) for t = 1:T_sim-1]
plt6 = plot!(t_sim[1:end-1],hcat(u_pfl_robust_conv...)[:],color=:cyan,label="robust pfl",linetype=:steppost)

plot(plt4,plt5,plt6,layout=(3,1))

# objective value
J_tvlqr_bnds
J_robust

# state tracking
Jx_tvlqr_bnds
Jx_robust

# control tracking
Ju_tvlqr_bnds
Ju_robust
