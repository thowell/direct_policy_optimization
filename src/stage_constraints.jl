function c_stage!(c,x,u,t)
	nothing
end

function stage_constraints!(c,Z,idx,T,m_stage)
	m_shift = 0
	for t = 1:T-1
		if m_stage[t] > 0
			x = Z[idx.x[t]]
			u = Z[idx.u[t]]
			c_stage!(view(c,m_shift .+ (1:m_stage[t])),x,u,t)
			m_shift += m_stage[t]
		end
	end
	nothing
end

function ∇stage_constraints!(∇c,Z,idx,T,m_stage)
	shift = 0
	m_shift = 0

	for t = 1:T-1
		if m_stage[t] > 0
			c_tmp = zeros(m_stage[t])

			x = view(Z,idx.x[t])
			u = view(Z,idx.u[t])
			con_x(c,z) = c_stage!(c,z,u,t)
			con_u(c,z) = c_stage!(c,x,z,t)

			r_idx = m_shift .+ (1:m_stage[t])

			c_idx = idx.x[t]
			len = length(r_idx)*length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(con_x,c_tmp,x))
			shift += len

			c_idx = idx.u[t]
			len = length(r_idx)*length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(con_u,c_tmp,u))
			shift += len

			m_shift += m_stage[t]
		end
	end
	nothing
end

function stage_constraint_sparsity(idx,T,m_stage;shift_r=0,shift_c=0)
	row = []
	col = []
	m_shift = 0
	for t = 1:T-1
		if m_stage[t] > 0
			r_idx = shift_r + m_shift .+ (1:m_stage[t])

			c_idx = shift_c .+ idx.x[t]
			row_col!(row,col,r_idx,c_idx)

			c_idx = shift_c .+ idx.u[t]
			row_col!(row,col,r_idx,c_idx)

			m_shift += m_stage[t]
		end
	end
	return collect(zip(row,col))
end
