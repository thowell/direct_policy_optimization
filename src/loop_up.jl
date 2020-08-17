function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx
	T = prob.T
	c[1:nx] = (Z[idx.x[2]] - Z[idx.x[T]])

	v1 = left_legendre(prob.model,Z[idx.x[2]],Z[idx.x[3]],Z[idx.u[1]],Z[idx.h[1]])
	vT = right_legendre(prob.model,Z[idx.x[T-1]],Z[idx.x[T]],Z[idx.u[T-2]],Z[idx.h[T-2]])

	c[nx .+ (1:nx)] = v1 - vT
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

	x2 = Z[idx.x[2]]
	x3 = Z[idx.x[3]]
	u1 = Z[idx.u[1]]
	h1 = Z[idx.h[1]]
	xT1 = Z[idx.x[T-1]]
	xT = Z[idx.x[T]]
	uT2 = Z[idx.u[T-2]]
	hT2 = Z[idx.h[T-2]]

	# v1(y) = legendre(prob.model,x2,x3,u1,h1)
	v1x2(y) = left_legendre(prob.model,y,x3,u1,h1)
	v1x3(y) = left_legendre(prob.model,x2,y,u1,h1)
	v1u1(y) = left_legendre(prob.model,x2,x3,y,h1)
	v1h1(y) = left_legendre(prob.model,x2,x3,u1,y)

	# vT(y) = legendre(prob.model,xT1,xT,uT2,hT2)
	vTxT1(y) = right_legendre(prob.model,y,xT,uT2,hT2)
	vTxT(y) = right_legendre(prob.model,xT1,y,uT2,hT2)
	vTuT2(y) = right_legendre(prob.model,xT1,xT,y,hT2)
	vThT2(y) = right_legendre(prob.model,xT1,xT,uT2,y)

	r_idx = nx .+ (1:nx)

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1x2,x2))
	shift += len

	c_idx = idx.x[3]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1x3,x3))
	shift += len

	c_idx = idx.u[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1u1,u1))
	shift += len

	c_idx = idx.h[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1h1,view(Z,idx.h[1])))
	shift += len

	c_idx = idx.x[T-1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vTxT1,xT1))
	shift += len

	c_idx = idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vTxT,xT))
	shift += len

	c_idx = idx.u[T-2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vTuT2,uT2))
	shift += len

	c_idx = idx.h[T-2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vThT2,view(Z,idx.h[T-2])))
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

		c_idx = idx.x[2]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[3]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.u[1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.h[1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[T-1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.x[T]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.u[T-2]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.h[T-2]
		row_col!(row,col,r_idx,c_idx)
	end
	return collect(zip(row,col))
end

# sample

function general_constraints!(c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	T = prob.prob.T

	m = nx

	for i = 1:prob.N
		c[(i-1)*(m+nx) .+ (1:m)] = (Z[idx_sample[i].x[2]] - Z[idx_sample[i].x[T]])

		v1 = left_legendre(prob.prob.model,Z[idx_sample[i].x[2]],Z[idx_sample[i].x[3]],Z[idx_sample[i].u[1]],Z[idx_sample[i].h[1]])
		vT = right_legendre(prob.prob.model,Z[idx_sample[i].x[T-1]],Z[idx_sample[i].x[T]],Z[idx_sample[i].u[T-2]],Z[idx_sample[i].h[T-2]])

		c[(i-1)*(m+nx) + m .+ (1:nx)] = v1 - vT
	end
	nothing
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	nx = prob.prob.nx
	idx_sample = prob.idx_sample
	T = prob.prob.T

	m = nx

	shift = 0

	for i = 1:prob.N
		r_idx = (i-1)*(m+nx) .+ (1:m)

		c_idx = idx_sample[i].x[2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(ones(m)))
		shift += len

		c_idx = idx_sample[i].x[T]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(m)))
		shift += len


		# c[nx-1 .+ (1:nx)] = v1 - vT
		r_idx = (i-1)*(m+nx) + m .+ (1:nx)

		x2 = Z[idx_sample[i].x[2]]
		x3 = Z[idx_sample[i].x[3]]
		u1 = Z[idx_sample[i].u[1]]
		h1 = Z[idx_sample[i].h[1]]
		xT1 = Z[idx_sample[i].x[T-1]]
		xT = Z[idx_sample[i].x[T]]
		uT2 = Z[idx_sample[i].u[T-2]]
		hT2 = Z[idx_sample[i].h[T-2]]

		# v1(y) = legendre(prob.model,x2,x3,u1,h1)
		v1x2(y) = left_legendre(prob.prob.model,y,x3,u1,h1)
		v1x3(y) = left_legendre(prob.prob.model,x2,y,u1,h1)
		v1u1(y) = left_legendre(prob.prob.model,x2,x3,y,h1)
		v1h1(y) = left_legendre(prob.prob.model,x2,x3,u1,y)

		# vT(y) = legendre(prob.model,xT1,xT,uT2,hT2)
		vTxT1(y) = right_legendre(prob.prob.model,y,xT,uT2,hT2)
		vTxT(y) = right_legendre(prob.prob.model,xT1,y,uT2,hT2)
		vTuT2(y) = right_legendre(prob.prob.model,xT1,xT,y,hT2)
		vThT2(y) = right_legendre(prob.prob.model,xT1,xT,uT2,y)

		c_idx = idx_sample[i].x[2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1x2,x2))
		shift += len

		c_idx = idx_sample[i].x[3]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1x3,x3))
		shift += len

		c_idx = idx_sample[i].u[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1u1,u1))
		shift += len

		c_idx = idx_sample[i].h[1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(v1h1,view(Z,idx_sample[i].h[1])))
		shift += len

		c_idx = idx_sample[i].x[T-1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vTxT1,xT1))
		shift += len

		c_idx = idx_sample[i].x[T]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vTxT,xT))
		shift += len

		c_idx = idx_sample[i].u[T-2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vTuT2,uT2))
		shift += len

		c_idx = idx_sample[i].h[T-2]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(-1.0*ForwardDiff.jacobian(vThT2,view(Z,idx_sample[i].h[T-2])))
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

		m = nx

		# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]
		for i = 1:prob.N
			r_idx = r_shift + (i-1)*(m+nx) .+ (1:m)

			c_idx = idx_sample[i].x[2]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T]
			row_col!(row,col,r_idx,c_idx)

			r_idx = r_shift + (i-1)*(m+nx) + m .+ (1:nx)

			c_idx = idx_sample[i].x[2]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[3]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].u[1]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].h[1]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T-1]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].x[T]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].u[T-2]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].h[T-2]
			row_col!(row,col,r_idx,c_idx)
		end
	end

	return collect(zip(row,col))
end
