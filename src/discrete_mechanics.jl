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

	0.5*h[1]*D1L1 + D2L1 + 0.5*h[1]*D1L2 - D2L2 + B_func(model,x3)'*u_ctrl + N_func(model,x3)'*λ + P_func(model,x3)'*b
end

function left_legendre(model,x1,x2,u,h)
	qm1 = (x1 + x2)/h[1]
	v1 = (x2 - x1)/h[1]

	u_ctrl = u[model.idx_u]
	λ = u[model.idx_λ]
	b = u[model.idx_b]

	D1L1,D2L1 = LagrangianDerivatives(model,qm1,v1)

	pl_del = -0.5*h[1]*D1L1 + D2L1 - 0.5*B_func(model,x2)'*u_ctrl - N_func(model,x2)'*λ - P_func(model,x2)'*b

	M_func(model,x2)\pl_del #TODO check which M(q...)
end

function right_legendre(model,x1,x2,u,h)
	qm1 = (x1 + x2)/h[1]
	v1 = (x2 - x1)/h[1]

	u_ctrl = u[model.idx_u]
	λ = u[model.idx_λ]
	b = u[model.idx_b]

	D1L1,D2L1 = LagrangianDerivatives(model,qm1,v1)

	pr_del = 0.5*h[1]*D1L1 + D2L1 + 0.5*B_func(model,x2)'u_ctrl

	M_func(model,x1)\pr_del #TODO check which M(q...)
end

# function discrete_dynamics(model,x1,x2,x3,u,h,t)
#     u_ctrl = u[model.idx_u]
#     λ = u[model.idx_λ]
#     b = u[model.idx_b]
#
#     (1/h[1])*(M_func(model,x1)*(x2 - x1) - M_func(model,x2)*(x3 - x2)) + h[1]*(0.5*C_func(model,x2,x3) - G_func(model,x2)) + transpose(B_func(model,x3))*u_ctrl + transpose(N_func(model,x3))*λ + transpose(P_func(model,x3))*b
# end
