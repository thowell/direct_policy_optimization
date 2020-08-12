function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx
	T = prob.T
	c[1:nx-1] = (Z[idx.x[3]] - Z[idx.x[T]])[2:end]

	v1 = (Z[idx.x[3]] - Z[idx.x[2]])/Z[idx.h[1]]
	vT = (Z[idx.x[T]] - Z[idx.x[T-1]])/Z[idx.h[T-2]]

	c[nx-1 .+ (1:nx)] = v1 - vT
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	shift = 0

	r_idx = 1:nx-1

	c_idx = idx.x[3][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(nx-1)))
	shift += len

	c_idx = idx.x[T][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx-1)))
	shift += len

	# v1 = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]
	# vT = (Z[idx.x[10]] - Z[idx.x[9]])/Z[idx.h[9]]
	#
	# c[nx-1 .+ (1:nx)] = v1 - vT
	r_idx = nx-1 .+ (1:nx)

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)./Z[idx.h[1]]))
	shift += len

	c_idx = idx.x[3]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(1.0*ones(nx)./Z[idx.h[1]]))
	shift += len

	c_idx = idx.h[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*(Z[idx.x[3]] - Z[idx.x[2]])/(Z[idx.h[1]]*Z[idx.h[1]]))
	shift += len

	c_idx = idx.x[T-1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(1.0*ones(nx)./Z[idx.h[T-2]]))
	shift += len

	c_idx = idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)./Z[idx.h[T-2]]))
	shift += len

	c_idx = idx.h[9]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(1.0*(Z[idx.x[T]] - Z[idx.x[T-1]])/(Z[idx.h[T-2]]*Z[idx.h[T-2]]))
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

	r_idx = r_shift .+ (1:nx-1)

	c_idx = idx.x[3][2:end]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[T][2:end]
	row_col!(row,col,r_idx,c_idx)

	r_idx = r_shift + nx-1 .+ (1:nx)

	c_idx = idx.x[2]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[3]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.h[1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[T-1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[T]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.h[T-2]
	row_col!(row,col,r_idx,c_idx)

	return collect(zip(row,col))
end
