function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx
	T = prob.T
	c[1:nx] = Z[idx.x[2]] - Z[idx.x[T]]
	c[nx .+ (1:nx)] = Z[idx.x[1]] - Z[idx.x[T-1]]
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	shift = 0

	r_idx = 1:nx

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(nx)))
	shift += len

	c_idx = idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
	shift += len

	c_idx = idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(nx)))
	shift += len

	c_idx = idx.x[T-1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
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

		# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]

		r_idx = r_shift .+ (1:nx)

		c_idx = idx.x[2]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[T]
		row_col!(row,col,r_idx,c_idx)

		r_idx = r_shift + nx .+ (1:nx)

		c_idx = idx.x[1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[T-1]
		row_col!(row,col,r_idx,c_idx)
	end
	return collect(zip(row,col))
end

# sample

function general_constraints!(c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	T = prob.prob.T

	for i = 1:prob.N
		c[(i-1)*(2*nx) .+ (1:nx)] = Z[idx_sample[i].x[2]] - Z[idx_sample[i].x[T]]
		c[(i-1)*(2*nx) + nx .+ (1:nx)] = Z[idx_sample[i].x[1]] - Z[idx_sample[i].x[T-1]]
	end

	nothing
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	T = prob.prob.T

	shift = 0

	for i = 1:prob.N
		r_idx = (i-1)*(2*nx) .+ (1:nx)

		c_idx = idx_sample[i].x[2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(ones(nx)))
		shift += len

		c_idx = idx_sample[i].x[T]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
		shift += len

		r_idx = (i-1)*(2*nx) + nx .+ (1:nx)

		c_idx = idx_sample[i].x[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(ones(nx)))
		shift += len

		c_idx = idx_sample[i].x[T-1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
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

		for i = 1:prob.N
			r_idx = r_shift + (i-1)*(2*nx) .+ (1:nx)

			c_idx = idx_sample[i].x[2]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T]
			row_col!(row,col,r_idx,c_idx)

			r_idx = r_shift + (i-1)*(2*nx) + nx .+ (1:nx)

			c_idx = idx_sample[i].x[1]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T-1]
			row_col!(row,col,r_idx,c_idx)
		end
	end

	return collect(zip(row,col))
end
