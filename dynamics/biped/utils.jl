include("f.jl")
include("g.jl")

function kinematics(model,q)
	θ1 = π - (q[1] + q[5])
	θ2 = π - (q[2] + q[5])

	pe = [model.l1*sin(θ1) + model.l2*sin(θ1 - q[3]),
		  model.l1*cos(θ1) + model.l2*cos(θ1 - q[3])]

	pf = [(model.l1*sin(θ2) + model.l2*sin(θ2 - q[4])),
		  pe[2] - (model.l1*cos(θ2) + model.l2*cos(θ2 - q[4]))]

    return pf
end

function pe(model,q)
	θ1 = π - (q[1] + q[5])
	θ2 = π - (q[2] + q[5])

	pe = [-1.0*(model.l1*sin(θ1) + model.l2*sin(θ1 - q[3])),
		  model.l1*cos(θ1) + model.l2*cos(θ1 - q[3])]

    return pe
end

function transformation_to_urdf_left_pinned(q,v)
	x = [q...,v...]
	robot_world_angle = (x[1] + x[3] + x[5]) - pi/2
	robot_world_angvel = (x[6] + x[8] + x[10])
	robot_position = [robot_world_angle; x[3]; -x[1]+pi/2; pi/2-x[2]; x[4]]
	robot_velocity = [robot_world_angvel; x[8]; -x[6] ; -x[7]; x[9]]
	return robot_position#, robot_velocity
end

function transformation_to_urdf_right_pinned(q,v)
	x = [q...,v...]
	robot_world_angle = (x[1] + x[3] + x[5]) - pi/2
	# robot_world_angvel = (x[6] + x[8] + x[10])
	robot_position = [robot_world_angle; -x[3]; x[1]-pi/2; x[2] - pi/2; -x[4]]
	# robot_velocity = [robot_world_angvel; x[8]; -x[6] ; -x[7]; x[9]]
	return robot_position#, robot_velocity
end

function Δ(x⁻)

    D_e_num = D_e(x⁻)
    E_2_num = E_2(x⁻)

	tmp = [Diagonal(ones(5)); zeros(2,5)]
    delta_F2 = -(E_2_num*(D_e_num\E_2_num'))\E_2_num*tmp
    delta_bar_dq_e = D_e_num\E_2_num'*delta_F2 + tmp

    R = [0 1 0 0 0; 1 0 0 0 0; 0 0 0 1 0; 0 0 1 0 0; 0 0 0 0 1] # relabelling matrix. Not be confused with a rotation matrix

    delta_dq = [R zeros(5,2)]*delta_bar_dq_e

    x⁺ = [R*x⁻[1:5]; delta_dq*x⁻[6:10]];
end

function D_e(in1)

	#    This function was generated by the Symbolic Math Toolbox version 8.0.
	#    10-Mar-2020 20:42:48

	q1 = in1[1]
	q2 = in1[2]
	q3 = in1[3]
	q4 = in1[4]
	q5 = in1[5]
	t2 = cos(q3);
	t3 = q1-q2+q3-q4;
	t4 = cos(t3);
	t5 = -q1+q2+q4;
	t6 = cos(t5);
	t7 = q1-q2;
	t8 = cos(t7);
	t9 = q1-q2+q3;
	t10 = cos(t9);
	t11 = t2.*1.12675666488;
	t12 = q1+q3+q5;
	t13 = q1+q5;
	t15 = t4.*1.35164722e-2;
	t16 = t10.*8.013743276e-2;
	t18 = t8.*7.1681372485e-2;
	t19 = t6.*1.2090221075e-2;
	t14 = -t15-t16-t18-t19;
	t17 = cos(q4);
	t20 = t17.*2.418044215e-2;
	t21 = q2+q4+q5;
	t22 = q2+q5;
	t23 = t2.*5.6337833244e-1;
	t24 = t23+7.02717990725e-1;
	t25 = -t15-t16;
	t26 = q1+q3;
	t27 = cos(t26);
	t28 = cos(t12);
	t29 = sin(t12);
	t30 = t29.*2.29192575;
	t31 = t17.*1.2090221075e-2;
	t32 = t31+1.0321331925e-2;
	t33 = cos(t21);
	t34 = t33.*4.388465e-2;
	t35 = sin(t21);
	t36 = cos(q1);
	t37 = -t15-t16-t18-t19+t20+7.8872363162e-2;
	t38 = -t15-t16+t23-t27.*5.57372816e-2+7.02717990725e-1;
	t39 = -t15-t19+t31+1.0321331925e-2;
	t40 = cos(t13);
	t41 = cos(t22);
	t42 = t41.*2.6018647e-1;
	t43 = sin(t13);
	t44 = t43.*1.82915043;
	t45 = sin(t22);
	t48 = t28.*2.29192575;
	t49 = t40.*1.82915043;
	t46 = -t48-t49;
	t47 = t34+t42;
	t50 = cos(q5);
	t51 = t50.*1.809652e-1;
	t52 = t30+t44;
	t54 = t35.*4.388465e-2;
	t55 = t45.*2.6018647e-1;
	t53 = -t54-t55;
	t56 = sin(q5);

	return reshape([t11+1.203518592942,t14,t24,-t15-t19,t11-t15-t16-t18-t19-t27.*5.57372816e-2-t36.*4.98559126e-2+1.203518592942,t46,t52,t14,t20+7.8872363162e-2,t25,t32,t37,t47,t53,t24,t25,7.02717990725e-1,-t15,t38,-t48,t30,t4.*(-1.35164722e-2)-t6.*1.2090221075e-2,t32,-t15,1.0321331925e-2,t39,t34,-t54,t4.*(-1.35164722e-2)-t6.*1.2090221075e-2-t8.*7.1681372485e-2-t10.*8.013743276e-2+t11-t27.*5.57372816e-2-t36.*4.98559126e-2+1.203518592942,t37,t38,t39,t4.*(-2.70329444e-2)-t6.*2.418044215e-2-t8.*1.4336274497e-1-t10.*1.6027486552e-1+t11+t20-t27.*1.114745632e-1-t36.*9.97118252e-2+1.313677579864,t34+t42-t48-t49+t51,t30+t44-t54-t55-t56.*1.809652e-1,t46,t47,t28.*(-2.29192575),t34,t28.*(-2.29192575)+t34-t40.*1.82915043+t42+t51,7.5838,0.0,t52,t53,t30,t35.*(-4.388465e-2),t30-t35.*4.388465e-2+t44-t45.*2.6018647e-1-t56.*1.809652e-1,0.0,7.5838],7,7)
end

function E_2(in1)

	#    This function was generated by the Symbolic Math Toolbox version 8.0.
	#    10-Mar-2020 20:42:50

	q1 = in1[1]
	q2 = in1[2]
	q3 = in1[3]
	q4 = in1[4]
	q5 = in1[5]
	t2 = q1+q3+q5;
	t3 = cos(t2);
	t4 = q2+q4+q5;
	t5 = cos(t4);
	t6 = t5.*(7.7e1./2.5e2);
	t7 = q1+q5;
	t8 = cos(t7);
	t9 = q2+q5;
	t10 = cos(t9);
	t11 = t10.*2.755e-1;
	t12 = sin(t2);
	t13 = t12.*(7.7e1./2.5e2);
	t14 = sin(t4);
	t15 = sin(t7);
	t16 = t15.*2.755e-1;
	t17 = sin(t9);

	return reshape([t3.*(-7.7e1./2.5e2)-t8.*2.755e-1,t13+t16,t6+t11,t14.*(-7.7e1./2.5e2)-t17.*2.755e-1,t3.*(-7.7e1./2.5e2),t13,t6,t14.*(-7.7e1./2.5e2),t3.*(-7.7e1./2.5e2)+t6-t8.*2.755e-1+t11,t13-t14.*(7.7e1./2.5e2)+t16-t17.*2.755e-1,1.0,0.0,0.0,1.0],2,7)
end
