# linear full-state-feedback policy
function policy(model,K,x,u,x_nom,u_nom)
	u_nom - reshape(K,model.nu_ctrl,model.nx)*(x - x_nom)
end

#TODO add velocity
