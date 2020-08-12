# linear full-state-feedback policy
function policy(model,K,x,u,x_nom,u_nom)
	u_nom - reshape(K,model.nu,model.nx)*(x - x_nom)
end

#TODO add velocity
