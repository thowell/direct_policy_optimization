function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx
	T = prob.T
	m = nx-1
	c[1:m] = (Z[idx.x[2]] - Z[idx.x[T]])[2:end]
	c[m .+ (1:m)] = (Z[idx.x[1]] - Z[idx.x[T-1]])[2:end]
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	shift = 0
	m = nx - 1
	r_idx = 1:m

	c_idx = idx.x[2][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(m)))
	shift += len

	c_idx = idx.x[T][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(m)))
	shift += len

	c_idx = idx.x[1][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(m)))
	shift += len

	c_idx = idx.x[T-1][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(m)))
	shift += len

	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0)

	row = []
	col = []
	if prob.general_constraints
		nx = prob.nx
		idx = prob.idx
		m = nx - 1

		# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]

		r_idx = r_shift .+ (1:m)

		c_idx = idx.x[2][2:end]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[T][2:end]
		row_col!(row,col,r_idx,c_idx)

		r_idx = r_shift + m .+ (1:m)

		c_idx = idx.x[1][2:end]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[T-1][2:end]
		row_col!(row,col,r_idx,c_idx)
	end
	return collect(zip(row,col))
end

# sample

function general_constraints!(c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	T = prob.prob.T

	m = nx-1

	for i = 1:prob.N
		c[(i-1)*(2m) .+ (1:m)] = (Z[idx_sample[i].x[2]] - Z[idx_sample[i].x[T]])[2:end]
		c[(i-1)*(2m) + m .+ (1:m)] = (Z[idx_sample[i].x[1]] - Z[idx_sample[i].x[T-1]])[2:end]
	end
	nothing
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	T = prob.prob.T

	m = nx-1

	shift = 0

	for i = 1:prob.N
		r_idx = (i-1)*(2*m) .+ (1:m)

		c_idx = idx_sample[i].x[2][2:end]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(ones(m)))
		shift += len

		c_idx = idx_sample[i].x[T][2:end]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(m)))
		shift += len

		r_idx = (i-1)*(2m) + m .+ (1:m)

		c_idx = idx_sample[i].x[1][2:end]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(ones(m)))
		shift += len

		c_idx = idx_sample[i].x[T-1][2:end]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(m)))
		shift += len

	end

	nothing
end

function general_constraint_sparsity(prob::SampleProblem;
		r_shift=0)

	row = []
	col = []

	if prob.sample_general_constraints
		nx = prob.prob.nx
		idx_sample = prob.idx_sample
		T = prob.prob.T

		m = nx-1

		# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]
		for i = 1:prob.N
			r_idx = r_shift + (i-1)*(2*m) .+ (1:m)

			c_idx = idx_sample[i].x[2][2:end]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T][2:end]
			row_col!(row,col,r_idx,c_idx)

			r_idx = r_shift + (i-1)*(2*m) + m .+ (1:m)

			c_idx = idx_sample[i].x[1][2:end]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T-1][2:end]
			row_col!(row,col,r_idx,c_idx)
		end
	end

	return collect(zip(row,col))
end
