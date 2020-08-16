function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx
	x2 = view(Z,idx.x[2])
	x3 = view(Z,idx.x[3])
	u1 = view(Z,idx.u[1])
	h1 = view(Z,idx.h[1])

	c[1:nx] = legendre(model,x2,x3,u1,h1) - prob.v1
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	x2 = view(Z,idx.x[2])
	x3 = view(Z,idx.x[3])
	u1 = view(Z,idx.u[1])
	h1 = view(Z,idx.h[1])

	lx2(y) = legendre(model,y,x3,u1,h1)
	lx3(y) = legendre(model,x2,y,u1,h1)
	lu1(y) = legendre(model,x2,x3,y,h1)
	lh1(y) = legendre(model,x2,x3,u1,y)

	shift = 0

	r_idx = 1:nx

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lx2,x2))
	shift += len

	c_idx = idx.x[3]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lx3,x3))
	shift += len

	c_idx = idx.u[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lu1,u1))
	shift += len

	c_idx = idx.h[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lh1,h1))
	shift += len

	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0)

	row = []
	col = []

	nx = prob.nx
	idx = prob.idx

	r_idx = r_shift .+ (1:nx)

	c_idx = idx.x[2]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[3]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.u[1]
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
		x2 = view(Z,idx_sample[i].x[2])
		x3 = view(Z,idx_sample[i].x[3])
		u1 = view(Z,idx_sample[i].u[1])
		h1 = view(Z,idx_sample[i].h[1])

		c[(i-1)*nx .+ (1:nx)] = legendre(model,x2,x3,u1,h1) - prob.prob.v1
	end
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	N = prob.N
	shift = 0

	for i = 1:N
		x2 = view(Z,idx_sample[i].x[2])
		x3 = view(Z,idx_sample[i].x[3])
		u1 = view(Z,idx_sample[i].u[1])
		h1 = view(Z,idx_sample[i].h[1])

		lx2(y) = legendre(model,y,x3,u1,h1)
		lx3(y) = legendre(model,x2,y,u1,h1)
		lu1(y) = legendre(model,x2,x3,y,h1)
		lh1(y) = legendre(model,x2,x3,u1,y)

		r_idx = (i-1)*nx .+ (1:nx)

		c_idx = idx_sample[i].x[2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lx2,x2))
		shift += len

		c_idx = idx_sample[i].x[3]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lx3,x3))
		shift += len

		c_idx = idx_sample[i].u[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lu1,u1))
		shift += len

		c_idx = idx_sample[i].h[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(lh1,h1))
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

		c_idx = idx_sample[i].x[2]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx_sample[i].x[3]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx_sample[i].u[1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx_sample[i].h[1]
		row_col!(row,col,r_idx,c_idx)
	end

	return collect(zip(row,col))
end
