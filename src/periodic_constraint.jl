function c_periodic!(c,x,u,t,model)
	nothing
end

function periodic_constraints!(c,Z,idx,T,model)
	nx = model.nx
	Tm = model.Tm

	x1 = Z[idx.x[1]]
	xTm = Z[idx.x[Tm]]
	xT = Z[idx.x[T]]

	c[1:nx] = x1 - Δ(model,xTm)
	c[nx .+ (1:nx)] = x1 - Δ(model,xT)
	nothing
end

function ∇periodic_constraints!(∇c,Z,idx,T,model)
	nx = model.nx
	Tm = model.Tm
	x1 = Z[idx.x[1]]
	xTm = Z[idx.x[Tm]]
	xT = Z[idx.x[T]]

	shift = 0

	Δ_tmp(z) = Δ(model,z)
	# c[1:nx] = x1 - Δ(model,xTm)
	r_idx = 1:nx

	c_idx = idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = Diagonal(ones(nx))
	shift += len

	c_idx = idx.x[Tm]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = -ForwardDiff.jacobian(Δ_tmp,xTm)
	shift += len

	# c[nx .+ (1:nx)] = x1 - Δ(model,xT)
	r_idx = nx .+ (1:nx)

	c_idx = idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = Diagonal(ones(nx))
	shift += len

	c_idx = idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = -ForwardDiff.jacobian(Δ_tmp,xT)
	shift += len

	nothing
end

function periodic_constraint_sparsity(idx,T,model;shift_r=0,shift_c=0)
	Tm = model.Tm
	
	row = []
	col = []

	shift = 0

	r_idx = shift_r .+ (1:nx)

	c_idx = shift_c .+ idx.x[1]
	# len = length(r_idx)*length(c_idx)
	# ∇c[shift .+ (1:len)] = Diagonal(ones(nx))
	# shift += len
	row_col!(row,col,r_idx,c_idx)

	c_idx = shift_c .+ idx.x[Tm]
	# len = length(r_idx)*length(c_idx)
	# ∇c[shift .+ (1:len)] = -ForwardDiff.jacobian(Δ_tmp,xTm)
	# shift += len
	row_col!(row,col,r_idx,c_idx)


	# c[nx .+ (1:nx)] = x1 - Δ(model,xT)
	r_idx = shift_r + nx .+ (1:nx)

	c_idx = shift_c .+ idx.x[1]
	# len = length(r_idx)*length(c_idx)
	# ∇c[shift .+ (1:len)] = Diagonal(ones(nx))
	# shift += len
	row_col!(row,col,r_idx,c_idx)

	c_idx = shift_c .+ idx.x[T]
	# len = length(r_idx)*length(c_idx)
	# ∇c[shift .+ (1:len)] = -ForwardDiff.jacobian(Δ_tmp,xT)
	# shift += len
	row_col!(row,col,r_idx,c_idx)

	return collect(zip(row,col))
end
