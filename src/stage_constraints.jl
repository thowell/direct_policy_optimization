function c_stage!(c,x,u,t,model)
	@error "stage constraints not defined"
	nothing
end

function c_stage!(c,x,t,model)
	@error " stage constraints not defined"
	nothing
end

function stage_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	idx = prob.idx
	T = prob.T
	m_stage = prob.m_stage
	model = prob.model

	m_shift = 0

	for (t,m) in enumerate(m_stage)
		if m > 0
			x = Z[idx.x[t]]
			u = Z[idx.u[t]]
			c_stage!(view(c,m_shift .+ (1:m)),x,u,t,model)
			m_shift += m
		end
	end
	nothing
end

function ∇stage_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	idx = prob.idx
	T = prob.T
	m_stage = prob.m_stage
	model = prob.model

	shift = 0
	m_shift = 0

	for (t,m) in enumerate(m_stage)
		if m > 0
			c_tmp = zeros(m)

			x = view(Z,idx.x[t])
			u = view(Z,idx.u[t])
			cx(c,z) = c_stage!(c,z,u,t,model)
			cu(c,z) = c_stage!(c,x,z,t,model)

			r_idx = m_shift .+ (1:m)

			c_idx = idx.x[t]
			len = length(r_idx)*length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cx,c_tmp,x))
			shift += len

			c_idx = idx.u[t]
			len = length(r_idx)*length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cu,c_tmp,u))
			shift += len

			m_shift += m
		end
	end

	nothing
end

function stage_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0)

	idx = prob.idx
	T = prob.T
	m_stage = prob.m_stage
	model = prob.model

	row = []
	col = []
	m_shift = 0

	for (t,m) in enumerate(m_stage)
		if m > 0
			r_idx = r_shift + m_shift .+ (1:m)

			c_idx = idx.x[t]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx.u[t]
			row_col!(row,col,r_idx,c_idx)

			m_shift += m
		end
	end

	return collect(zip(row,col))
end
