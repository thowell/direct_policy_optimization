function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.model.nx
	T = prob.T
	c[1:nx] = Z[prob.idx.x[1]] - Z[prob.idx.x[T]]
	nothing
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.model.nx
	T = prob.T

	s = 0
	# c[1:nx] = Z[prob.idx.x[1]] - Z[prob.idx.x[T]]

	r_idx = (1:nx)

	c_idx = prob.idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[s .+ (1:len)] = vec(Diagonal(ones(nx)))
	s += len

	c_idx = prob.idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[s .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)))
	s += len

	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0)
	row = []
	col = []

	nx = prob.model.nx
	T = prob.T

	r_idx = r_shift .+ (1:nx)

	c_idx = prob.idx.x[1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = prob.idx.x[T]
	row_col!(row,col,r_idx,c_idx)

	return collect(zip(row,col))
end
