# function simulate_linear_controller(Kc,z_nom,u_nom,model,Q,R,T_sim,Δt,z0,w;
#         _norm=2,
#         ul=-Inf*ones(length(u_nom[1])),
#         uu=Inf*ones(length(u_nom[1])))
#     T = length(Kc)+1
#     times = [(t-1)*Δt for t = 1:T-1]
#     tf = Δt*T
#     t_sim = range(0,stop=tf,length=T_sim)
#     dt_sim = tf/(T_sim-1)
#
#     z_rollout = [z0]
#     u_rollout = []
#     J = 0.0
#     Jx = 0.0
#     Ju = 0.0
#     for tt = 1:T_sim-1
#         t = t_sim[tt]
#         k = searchsortedlast(times,t)
#         z = z_rollout[end] + dt_sim*w[:,tt]
#         u = u_nom[k] - Kc[k]*(z - z_nom[k])
#
#         # clip controls
#         u = max.(u,ul)
#         u = min.(u,uu)
#
#         push!(z_rollout,rk3(model,z,u,dt_sim))
#         push!(u_rollout,u)
#         if _norm == 2
#             J += (z_rollout[end]-z_nom[k+1])'*Q[k+1]*(z_rollout[end]-z_nom[k+1])
#             J += (u_rollout[end]-u_nom[k])'*R[k]*(u_rollout[end]-u_nom[k])
#             Jx += (z_rollout[end]-z_nom[k+1])'*Q[k+1]*(z_rollout[end]-z_nom[k+1])
#             Ju += (u_rollout[end]-u_nom[k])'*R[k]*(u_rollout[end]-u_nom[k])
#         else
#             J += norm(sqrt(Q[k+1])*(z_rollout[end]-z_nom[k+1]),_norm)
#             J += norm(sqrt(R[k])*(u-u_nom[k]),_norm)
#             Jx += norm(sqrt(Q[k+1])*(z_rollout[end]-z_nom[k+1]),_norm)
#             Ju += norm(sqrt(R[k])*(u-u_nom[k]),_norm)
#         end
#     end
#     return z_rollout, u_rollout, J/(T_sim-1), Jx/(T_sim-1), Ju/(T_sim-1)
# end
#
# function nominal_trajectories(z_nom,u_nom,T_sim,Δt)
#     T = length(z_nom)
#     times = [(t-1)*Δt for t = 1:T-1]
#     tf = Δt*T
#     t_sim = range(0,stop=tf,length=T_sim)
#     dt_sim = tf/(T_sim-1)
#
#     _z_nom = [z_nom[1]]
#     _u_nom = []
#     for tt = 1:T_sim-1
#         t = t_sim[tt]
#         k = searchsortedlast(times,t)
#
#         push!(_z_nom,z_nom[k])
#         push!(_u_nom,u_nom[k])
#     end
#     return _z_nom, _u_nom
# end

function simulate(model,xpp,xp,dt_sim,tf;
		tol=1.0e-6,c_tol=1.0e-6,α=100.0,slack_tol=1.0e-5)

	# time
    T_sim = floor(convert(Int,tf/dt_sim)) + 1

	# Bounds

	# ul <= u <= uu
	uu_sim = Inf*ones(model.nu)
	uu_sim[model.idx_u] .= 0.0
	ul_sim = zeros(model.nu)
	ul_sim[model.idx_u] .= 0.0

	# h = h0 (fixed timestep)
	hu_sim = dt_sim
	hl_sim = dt_sim

	model.α = α
	penalty_obj = PenaltyObjective(model.α)
	multi_obj = MultiObjective([penalty_obj])

	X_traj = [xpp,xp]
	U_traj = []

	for t = 1:T_sim
		# xl <= x <= xu
		xu_sim = [X_traj[t],X_traj[t+1],Inf*ones(model.nx)]
		xl_sim = [X_traj[t],X_traj[t+1],-Inf*ones(model.nx)]

		# Problem
		prob_sim = init_problem(model.nx,model.nu,3,model,multi_obj,
		                    xl=xl_sim,
		                    xu=xu_sim,
		                    ul=[ul_sim],
		                    uu=[uu_sim],
		                    hl=[dt_sim],
		                    hu=[dt_sim]
		                    )
		# MathOptInterface problem
		prob_sim_moi = init_MOI_Problem(prob_sim)

		# Pack trajectories into vector
		Z0_sim = pack([X_traj[t],X_traj[t+1],X_traj[t+1]],[t == 1 ? rand(model.nu) : U_traj[t-1]],dt_sim,prob_sim)

		@time Z_sim_sol = solve(prob_sim_moi,copy(Z0_sim),tol=tol,c_tol=c_tol)
		X_sol, U_sol, H_sol = unpack(Z_sim_sol,prob_sim)

		@assert U_sol[1][model.idx_s] < slack_tol

		push!(X_traj,X_sol[end])
		push!(U_traj,U_sol[1])
	end
	return X_traj, U_traj
end

function simulate_policy(model,X_nom,U_nom,H_nom,K_nom,T_sim;
		tol=1.0e-6,c_tol=1.0e-6,α=100.0,slack_tol=1.0e-5)

	times = [(t-1)*H_nom[t] for t = 1:T-2]
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

	# Bounds

	# ul <= u <= uu
	uu_sim = Inf*ones(model.nu)
	uu_sim[model.idx_u] .= Inf
	ul_sim = zeros(model.nu)
	ul_sim[model.idx_u] .= -Inf

	# h = h0 (fixed timestep)
	hu_sim = dt_sim
	hl_sim = dt_sim

	model.α = α
	penalty_obj = PenaltyObjective(model.α)
	multi_obj = MultiObjective([penalty_obj])

	X_traj = [X_nom[1],X_nom[2]]
	U_traj = []

	for t = 1:T_sim

	    k = searchsortedlast(times,t_sim[t])

		# xl <= x <= xu
		xu_sim = [X_traj[t],X_traj[t+1],Inf*ones(model.nx)]
		xl_sim = [X_traj[t],X_traj[t+1],-Inf*ones(model.nx)]

		# policy
		pi = PolicyInfo(X_nom[k:k+2],U_nom[k:k],H_nom[k:k],K_nom[k:k],dt_sim)

		# Problem
		prob_sim = init_problem(model.nx,model.nu,3,model,multi_obj,
			                    xl=xl_sim,
			                    xu=xu_sim,
			                    ul=[ul_sim],
			                    uu=[uu_sim],
			                    hl=[dt_sim],
			                    hu=[dt_sim],
								general_constraints=true,
								m_general=model.nu_ctrl,
								general_ineq=(1:0),
			                    policy_info=pi)

		# MathOptInterface problem
		prob_sim_moi = init_MOI_Problem(prob_sim)

		# Pack trajectories into vector
		Z0_sim = pack([X_traj[t],X_traj[t+1],X_traj[t+1]],[t == 1 ? U_nom[1] : U_traj[t-1]],dt_sim,prob_sim)

		@time Z_sim_sol = solve(prob_sim_moi,copy(Z0_sim),tol=tol,c_tol=c_tol)
		X_sol, U_sol, H_sol = unpack(Z_sim_sol,prob_sim)

		# @assert U_sol[1][model.idx_s] < slack_tol

		push!(X_traj,X_sol[end])
		push!(U_traj,U_sol[1])
	end
	return X_traj, U_traj, dt_sim
end

function general_constraints!(c,z,prob::TrajectoryOptimizationProblem)
   	nx = prob.nx
    nu = prob.nu
	u_policy = prob.model.idx_u
    nu_policy = prob.model.nu_ctrl
	idx = prob.idx
	pi = prob.policy_info

    # controller for samples
    x1 = view(z,idx.x[1])
    x2 = view(z,idx.x[2])
    x3 = view(z,idx.x[3])
    u = view(z,idx.u[1][prob.model.idx_u])
    ū = view(z,idx.u[1][(nu_policy+1):nu])
    h = view(z,idx.h[1])
    c[(1:nu_policy)] = policy(model,
		pi.K[1],x1,x2,x3,ū,pi.Δt,
		pi.X[1],pi.X[2],pi.X[3],
		pi.U[1][prob.model.idx_u],pi.U[1][prob.model.nu_ctrl+1:prob.model.nu],pi.Δt) - u

    nothing
end

function ∇general_constraints!(∇c,z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
    nu = prob.nu
	u_policy = prob.model.idx_u
    nu_policy = prob.model.nu_ctrl
	idx = prob.idx
	pi = prob.policy_info

    # controller for samples
    x1 = view(z,idx.x[1])
    x2 = view(z,idx.x[2])
    x3 = view(z,idx.x[3])
    u = view(z,idx.u[1][u_policy])
    ū = view(z,idx.u[1][(nu_policy+1):nu])

    px1(y) = policy(model,pi.K[1],y,x2,x3,ū,pi.Δt,pi.X[1],pi.X[2],pi.X[3],pi.U[1][prob.model.idx_u],pi.U[1][prob.model.nu_ctrl+1:prob.model.nu],pi.Δt)
    px2(y) = policy(model,pi.K[1],x1,y,x3,ū,pi.Δt,pi.X[1],pi.X[2],pi.X[3],pi.U[1][prob.model.idx_u],pi.U[1][prob.model.nu_ctrl+1:prob.model.nu],pi.Δt)
    px3(y) = policy(model,pi.K[1],x1,x2,y,ū,pi.Δt,pi.X[1],pi.X[2],pi.X[3],pi.U[1][prob.model.idx_u],pi.U[1][prob.model.nu_ctrl+1:prob.model.nu],pi.Δt)
    pū(y) = policy(model,pi.K[1],x1,x2,x3,y,pi.Δt,pi.X[1],pi.X[2],pi.X[3],pi.U[1][prob.model.idx_u],pi.U[1][prob.model.nu_ctrl+1:prob.model.nu],pi.Δt)

	s = 0

    r_idx = (1:nu_policy)

    c_idx = idx.x[1]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1,x1))
    s += len

    c_idx = idx.x[2]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2,x2))
    s += len

    c_idx = idx.x[3]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3,x3))
    s += len

    c_idx = idx.u[1][(nu_policy+1):nu]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pū,ū))
    s += len

    c_idx = idx.u[1][u_policy]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nu_policy)))
    s += len

    nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
    r_shift=0)
	nx = prob.nx
    nu = prob.nu
	u_policy = prob.model.idx_u
    nu_policy = prob.model.nu_ctrl
	idx = prob.idx
	pi = prob.policy_info

	row = []
	col = []
    # controller for samples

    r_idx = r_shift .+ (1:nu_policy)

    c_idx = idx.x[1]
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.x[2]
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.x[3]
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.u[1][(nu_policy+1):nu]
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.u[1][u_policy]
	row_col!(row,col,r_idx,c_idx)

    return collect(zip(row,col))
end
