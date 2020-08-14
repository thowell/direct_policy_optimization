struct PolicyInfo
	X
	U
	H
	K
end

PolicyInfo() = PolicyInfo([],[],[],[])

# linear full-state-feedback policy
function policy(model,K,x1,x2,x3,u,h,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
	u_nom[model.idx_u] - reshape(K,model.nu_ctrl,model.nx)*(x3 - x3_nom)
end
