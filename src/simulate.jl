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

function simulate_policy(model,X_nom,U_nom,K_nom,dt_sim,tf;
		tol=1.0e-6,c_tol=1.0e-6,α=100.0,slack_tol=1.0e-5)

    T_sim = floor(convert(Int,tf/dt_sim)) + 1

	# Bounds

	# ul <= u <= uu
	uu_sim = Inf*ones(model.nu)
	# uu_sim[model.idx_u] .= 0.0
	ul_sim = zeros(model.nu)
	# ul_sim[model.idx_u] .= 0.0

	# h = h0 (fixed timestep)
	hu_sim = dt_sim
	hl_sim = dt_sim

	model.α = α
	penalty_obj = PenaltyObjective(model.α)
	multi_obj = MultiObjective([penalty_obj])

	X_traj = [x_nom[1],x_nom[2]]
	U_traj = []

	for t = 1:T_sim
		x1_nom = X_nom[t]
		x2_nom = X_nom[t+1]
		x3_nom = X_nom[t+2]
		u_nom = U_nom[t]

		K = K_nom[t]
		function general_constraints!(c,z,prob::TrajectoryOptimizationProblem)
		   	nx = prob.nx
		    nu = prob.nu
		    nu_policy = prob.model.nu_ctrl

		    # controller for samples
		    x1 = view(z,idx.x[t])
		    x2 = view(z,idx.x[t+1])
		    x3 = view(z,idx.x[t+2])
		    u = view(z,idx.u[t][prob.model.idx_u])
		    ū = view(z,idx.u[t][(nu_policy+1):nu])
		    h = view(z,idx.h[t])
		    c[(1:nu_policy)] = policy(model,
				K,x1,x2,x3,ū,dt_sim,
				x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim) - u

		    nothing
		end

		function ∇general_constraints!(∇c,z,prob::TrajectoryOptimizationProblem)
			nx = prob.nx
		    nu = prob.nu
		    nu_policy = prob.model.nu_ctrl

		    # controller for samples
		    x1 = view(z,idx.x[t])
		    x2 = view(z,idx.x[t+1])
		    x3 = view(z,idx.x[t+2])
		    u = view(z,idx.u[t][u_policy])
		    ū = view(z,idx.u[t][(nu_policy+1):nu])
		    c[(1:nu_policy)] = policy(model,K,x1,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim) - u

		    px1(y) = policy(model,K,y,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
		    px2(y) = policy(model,K,x1,y,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
		    px3(y) = policy(model,K,x1,x2,y,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
		    pūi(y) = policy(model,K,x1,x2,x3,y,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)

		    r_idx = (1:nu_policy)

		    c_idx = idx.x[t]
		    len = length(r_idx)*length(c_idx)
		    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1,x1))
		    s += len

		    c_idx = idx.x[t+1]
		    len = length(r_idx)*length(c_idx)
		    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2,x2))
		    s += len

		    c_idx = idx.x[t+2]
		    len = length(r_idx)*length(c_idx)
		    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3,x3))
		    s += len

		    c_idx = idx.u[t][(nu_policy+1):nu]
		    len = length(r_idx)*length(c_idx)
		    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pū,ū))
		    s += len

		    c_idx = idx.u[t][u_policy]
		    len = length(r_idx)*length(c_idx)
		    ∇c[s .+ (1:len)] = vec(-Im)
		    s += len

		    nothing
		end

		function general_constraints_sparsity(prob::TrajectoryOptimizationProblem;
		    r_shift=0)
			nx = prob.nx
		    nu = prob.nu
		    nu_policy = prob.model.nu_ctrl

		    # # controller for samples
		    # x1 = view(z,idx.x[t])
		    # x2 = view(z,idx.x[t+1])
		    # x3 = view(z,idx.x[t+2])
		    # u = view(z,idx.u[t][u_policy])
		    # ū = view(z,idx.u[t][(nu_policy+1):nu])
		    # c[(1:nu_policy)] = policy(model,K,x1,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim) - u
			#
		    # px1(y) = policy(model,K,y,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
		    # px2(y) = policy(model,K,x1,y,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
		    # px3(y) = policy(model,K,x1,x2,y,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
		    # pūi(y) = policy(model,K,x1,x2,x3,y,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)

		    r_idx = r_shift .+ (1:nu_policy)

		    c_idx = idx.x[t]
		    # len = length(r_idx)*length(c_idx)
		    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1,x1))
		    # s += len
			row_col!(row,col,r_idx,c_idx)

		    c_idx = idx.x[t+1]
		    # len = length(r_idx)*length(c_idx)
		    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2,x2))
		    # s += len
			row_col!(row,col,r_idx,c_idx)

		    c_idx = idx.x[t+2]
		    # len = length(r_idx)*length(c_idx)
		    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3,x3))
		    # s += len
			row_col!(row,col,r_idx,c_idx)

		    c_idx = idx.u[t][(nu_policy+1):nu]
		    # len = length(r_idx)*length(c_idx)
		    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pū,ū))
		    # s += len
			row_col!(row,col,r_idx,c_idx)

		    c_idx = idx.u[t][u_policy]
		    # len = length(r_idx)*length(c_idx)
		    # ∇c[s .+ (1:len)] = vec(-Im)
		    # s += len
			row_col!(row,col,r_idx,c_idx)

		    return collect(zip(row,col))
		end

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
		                    hu=[dt_sim],
							general_constraints=true,
							m_general=model.nu_ctrl,
							general_ineq=(1:0)
		                    )
		# MathOptInterface problem
		prob_sim_moi = init_MOI_Problem(prob_sim)

		# Pack trajectories into vector
		Z0_sim = pack([X_traj[t],X_traj[t+1],X_traj[t+1]],[t == 1 ? U_nom[1] : U_traj[t-1]],dt_sim,prob_sim)

		@time Z_sim_sol = solve(prob_sim_moi,copy(Z0_sim),tol=tol,c_tol=c_tol)
		X_sol, U_sol, H_sol = unpack(Z_sim_sol,prob_sim)

		@assert U_sol[1][model.idx_s] < slack_tol

		push!(X_traj,X_sol[end])
		push!(U_traj,U_sol[1])
	end
	return X_traj, U_traj
end

function general_constraints!(c,z,prob::TrajectoryOptimizationProblem)
   	nx = prob.nx
    nu = prob.nu
    nu_policy = prob.model.nu_ctrl

    # controller for samples
    x1 = view(z,idx.x[t])
    x2 = view(z,idx.x[t+1])
    x3 = view(z,idx.x[t+2])
    u = view(z,idx.u[t][prob.model.idx_u])
    ū = view(z,idx.u[t][(nu_policy+1):nu])
    h = view(z,idx.h[t])
    c[(1:nu_policy)] = policy(model,
		K,x1,x2,x3,ū,dt_sim,
		x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim) - u

    nothing
end

function ∇general_constraints!(∇c,z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
    nu = prob.nu
    nu_policy = prob.model.nu_ctrl

    # controller for samples
    x1 = view(z,idx.x[t])
    x2 = view(z,idx.x[t+1])
    x3 = view(z,idx.x[t+2])
    u = view(z,idx.u[t][u_policy])
    ū = view(z,idx.u[t][(nu_policy+1):nu])
    c[(1:nu_policy)] = policy(model,K,x1,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim) - u

    px1(y) = policy(model,K,y,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
    px2(y) = policy(model,K,x1,y,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
    px3(y) = policy(model,K,x1,x2,y,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
    pūi(y) = policy(model,K,x1,x2,x3,y,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)

    r_idx = (1:nu_policy)

    c_idx = idx.x[t]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1,x1))
    s += len

    c_idx = idx.x[t+1]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2,x2))
    s += len

    c_idx = idx.x[t+2]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3,x3))
    s += len

    c_idx = idx.u[t][(nu_policy+1):nu]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pū,ū))
    s += len

    c_idx = idx.u[t][u_policy]
    len = length(r_idx)*length(c_idx)
    ∇c[s .+ (1:len)] = vec(-Im)
    s += len

    nothing
end

function general_constraints_sparsity(prob::TrajectoryOptimizationProblem;
    r_shift=0)
	nx = prob.nx
    nu = prob.nu
    nu_policy = prob.model.nu_ctrl

    # controller for samples
    x1 = view(z,idx.x[t])
    x2 = view(z,idx.x[t+1])
    x3 = view(z,idx.x[t+2])
    u = view(z,idx.u[t][u_policy])
    ū = view(z,idx.u[t][(nu_policy+1):nu])
    c[(1:nu_policy)] = policy(model,K,x1,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim) - u

    px1(y) = policy(model,K,y,x2,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
    px2(y) = policy(model,K,x1,y,x3,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
    px3(y) = policy(model,K,x1,x2,y,ū,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)
    pūi(y) = policy(model,K,x1,x2,x3,y,dt_sim,x1_nom,x2_nom,x3_nom,u_nom[prob.model.idx_u],u_nom[prob.model.nu_ctrl+1:prob.model.nu],dt_sim)

    r_idx = r_shift .+ (1:nu_policy)

    c_idx = idx.x[t]
    # len = length(r_idx)*length(c_idx)
    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px1,x1))
    # s += len
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.x[t+1]
    # len = length(r_idx)*length(c_idx)
    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px2,x2))
    # s += len
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.x[t+2]
    # len = length(r_idx)*length(c_idx)
    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(px3,x3))
    # s += len
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.u[t][(nu_policy+1):nu]
    # len = length(r_idx)*length(c_idx)
    # ∇c[s .+ (1:len)] = vec(ForwardDiff.jacobian(pū,ū))
    # s += len
	row_col!(row,col,r_idx,c_idx)

    c_idx = idx.u[t][u_policy]
    # len = length(r_idx)*length(c_idx)
    # ∇c[s .+ (1:len)] = vec(-Im)
    # s += len
	row_col!(row,col,r_idx,c_idx)

    return collect(zip(row,col))
end
