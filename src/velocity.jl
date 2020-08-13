function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]] - [1.0;0.0;1.0]
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	shift = 0
	# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]

	r_idx = 1:nx

	c_idx = idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0/Z[idx.h[1]]*ones(nx)))
	shift += len

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(1.0/Z[idx.h[1]]*ones(nx)))
	shift += len

	c_idx = idx.h[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*(Z[idx.x[2]] - Z[idx.x[1]])/(Z[idx.h[1]]*Z[idx.h[1]]))
	shift += len

	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0)

	row = []
	col = []

	nx = prob.nx
	idx = prob.idx

	# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]

	r_idx = r_shift .+ (1:nx)

	c_idx = idx.x[1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[2]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.h[1]
	row_col!(row,col,r_idx,c_idx)

	return collect(zip(row,col))
end

function general_constraints!(c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	N = prob.N

	for i = 1:N
		c[(i-1)*nx .+ (1:nx)] = (Z[idx_sample[i].x[2]] - Z[idx_sample[i].x[1]])/Z[idx_sample[i].h[1]] #- [1.0;0.0;0.0]
	end
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	N = prob.N
	shift = 0

	for i = 1:N
		r_idx = (i-1)*nx .+ (1:nx)

		c_idx = idx_sample[i].x[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0/Z[idx_sample[i].h[1]]*ones(nx)))
		shift += len

		c_idx = idx_sample[i].x[2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(1.0/Z[idx_sample[i].h[1]]*ones(nx)))
		shift += len

		c_idx = idx_sample[i].h[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(-1.0*(Z[idx_sample[i].x[2]] - Z[idx_sample[i].x[1]])/(Z[idx_sample[i].h[1]]*Z[idx_sample[i].h[1]]))
		shift += len
	end

	nothing
end

function general_constraint_sparsity(prob::SampleProblem;
		r_shift=0)

	row = []
	col = []

	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	N = prob.N

	for i = 1:N
		r_idx = r_shift + (i-1)*nx .+ (1:nx)

		c_idx = idx_sample[i].x[1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx_sample[i].x[2]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx_sample[i].h[1]
		row_col!(row,col,r_idx,c_idx)
	end

	return collect(zip(row,col))
end
