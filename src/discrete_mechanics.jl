function LagrangianDerivatives(model,q,v)
	D1L = 0.5*C_func(model,q,v) - G_func(model,q)
	D2L = M_func(model,q)*v

	return D1L, D2L
end

function discrete_dynamics(model,x1,x2,x3,u,h,t)
	qm1 = (x1 + x2)/h[1]
	qm2 = (x2 + x3)/h[1]
	v1 = (x2-x1)/h[1]
	v2 = (x3-x2)/h[1]

	u_ctrl = u[model.idx_u]
	λ = u[model.idx_λ]
	b = u[model.idx_b]

	D1L1, D2L1 = LagrangianDerivatives(model,qm1,v1)
	D1L2, D2L2 = LagrangianDerivatives(model,qm2,v2)

	0.5*h[1]*D1L1 + D2L1 + 0.5*h[1]*D1L2 - D2L2 + h[1]*B_func(model,x3)'*u_ctrl + h[1]*N_func(model,x3)'*λ + h[1]*P_func(model,x3)'*b
end

function legendre(model,x1,x2,u,h)
	qm1 = (x1 + x2)/h[1]
	v1 = (x2 - x1)/h[1]

	u_ctrl = u[model.idx_u]
	λ = u[model.idx_λ]
	b = u[model.idx_b]

	D1L1,D2L1 = LagrangianDerivatives(model,qm1,v1)

	pl_del = -0.5*h[1]*D1L1 + D2L1 - 0.5*h[1]*B_func(model,x2)'u_ctrl - 0.5*h[1]*B_func(model,x2)'*u_ctrl - h[1]*N_func(model,x2)'*λ - h[1]*P_func(model,x2)'*b

	M_func(model,x2)\pl_del
end
