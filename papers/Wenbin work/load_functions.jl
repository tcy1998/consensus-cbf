
eye(x) = Matrix{Float64}(I, x, x)
f(xa, x)=xa[3]/((x[1]-xa[1])^2+(x[2]-xa[2])^2)
function circleShape(h, k, r)
    theta = LinRange(0, 2*pi, 500)
    h .+ r * sin.(theta), k .+ r*cos.(theta)
end

function attacker(x_safe,x_k,xa)
    dis = sqrt((x_k[1]-xa[1])^2+(x_k[2]-xa[2])^2)
    if dis >= x_safe
        d_k = zeros(2,1);
        at_mode = 0
    else
        # d_k = [100; 100];
        d_k = [10; 10];
        at_mode = 1
    end

    return  d_k, at_mode, dis
end

function agent(x_km1,x_k,u_k,d_k,P_km1,x_hat_km1,at_mode)
    v_max = 5;

    sig_w = eye(4)   * 0.01; #UAV 0.01
    sig_v_G = eye(2) * 1; #GPS 1
    sig_v_I = eye(2) * 0.2; #IMU 0.01->0.1
    sig_v_A =   0.1; # Signal strength 0.001

    A = [1 0 0.1 0; 0 1 0 0.1; 0 0 1 0; 0 0 0 1] # matrix 4-by-4
    B = [0 0; 0 0; 0.1 0; 0 0.1] # matrix 4-by-2
    C_G = [1 0 0 0; 0 1 0 0] # matrix 2-by-4
    C_I = [0 0 1 0; 0 0 0 1];# matrix 2-by-4
    C_A = 1; # matrix 1-by-1

    w = sig_w * randn(4,1);
    v_G = sig_v_G * randn(2,1);
    v_I = sig_v_I * randn(2,1);
    v_A = sig_v_A  * randn(1,1);

    x_kp1 = A * x_k + B * u_k + w
    y_G_k = C_G * x_k + d_k + v_G
    y_I_k = C_I * (x_k - x_km1) + v_I
    y_S_k = 0#f(xxa, x_k)

    # if x_kp1[3]^2 + x_kp1[4]^2 > v_max^2
    #     x_kp1[3:4] = v_max * x_kp1[3:4]/(sqrt(x_kp1[3:4]' * x_kp1[3:4]))
    # end
    """
    ISE
    """
    C = [C_G; C_I]; #4-by-4
    D = [zeros(2,2) zeros(2,2); zeros(2,2) eye(2)];  #4-by-4
    sig_y = [sig_v_G zeros(2,2); zeros(2,2) sig_v_I]; #4-by-4
    # gain calculation
    K = (A * P_km1 * (C*A-D*C)' + sig_w * C')* inv((C*A-D*C)*P_km1*(C*A-D*C)' + C*sig_w*C' + sig_y);
    K_I =K[:,3:4]; #4-by-2

    if at_mode == 1
        # gain renew
        K_G = zeros(4,2);
    else
        K_G = K[:,1:2]; #4-by-2
    end

    # output 1
    x_hat_k = A * x_hat_km1 + B*u_k + K_G * (y_G_k - C_G * (A * x_hat_km1 + B * u_k)) + K_I * (y_I_k - C_I * (A * x_hat_km1 + B * u_k - x_hat_km1));

    # output 2
    P_k = (A - K*C*A + K*D*C)*P_km1* (A - K*C*A + K*D*C)' + (eye(4)- K*C)* sig_w *(eye(4)- K*C)' + K * sig_y * K';

    # output 3
    d_hat_k = y_G_k - C_G*(A * x_hat_km1 + B* u_k);

    # output 4
    P_d_k = C_G * (A * P_km1 * A' + sig_w)* C_G' + sig_v_G;


    return x_kp1, x_hat_k, P_k, d_hat_k
end

function PD_controller(x_goal, x_hat_km1,x_k)
    u_max = 2;
    v_max = 5;
    A = [1 0 0.1 0; 0 1 0 0.1; 0 0 1 0; 0 0 0 1] # matrix 4-by-4
    B = [0 0; 0 0; 0.1 0; 0 0.1] # matrix 4-by-2
    # PD control
    Pro = 0.05
    Der = 0.56
    PDc = [Pro 0  Der 0; 0 Pro 0  Der]
    u_k = PDc * (x_goal - x_hat_km1);

    if u_k[1]^2 + u_k[2]^2 > u_max^2
        u_k = u_max * u_k/(sqrt(u_k' * u_k))
    end

    # Speed limits in 2-D
    x_pp = A * x_k + B * u_k
    if x_pp[3]^2 + x_pp[4]^2 > v_max^2
        vo_k = [x_pp[3];x_pp[4]]
        vo_k = v_max * vo_k/(sqrt(vo_k' * vo_k))
        x_pp[3] = vo_k[1]
        x_pp[4] = vo_k[2]
        u_k[1] = x_pp[3] - x_k[3]
        u_k[2] = x_pp[4] - x_k[4]
    end

    # if x_hat_km1[3]^2 + x_hat_km1[4]^2 > v_max^2
    #     u_k = [0;0];
    # end


    return u_k
end

function MPC_PF(x_hat_km1,xa,k_it,k_a,k_esc_cal,x_safe,x_goal)
    # x_hat_km1: previous UAV state 4x1
    # xa: attacker state 4x1
    # k_it: current time step 1x1
    # k_a: attack time 1x1
    # x_safe: safe distance to the spoofer 1x1
    # x_goal: UAV goal state 4x1
    u_max = 2;
    v_max = 5;
    A = [1 0 0.1 0; 0 1 0 0.1; 0 0 1 0; 0 0 0 1] # matrix 4-by-4
    B = [0 0; 0 0; 0.1 0; 0 0.1] # matrix 4-by-2
    N = round(k_esc_cal)+20 #+20

    """transfm"""
    xt = x_hat_km1
    X_safe = x_safe*ones(N)

    X_goal = x_goal
    for i= 2:N
        X_goal = [X_goal;x_goal]
    end

    """optimizaion"""
    model = Model(with_optimizer(Ipopt.Optimizer,print_level=0))
    @variable(model, -2 <= U[1:2N] <= 2)
    @variable(model, Xtt[1:4N-4])
    @expression(model, Xt, [xt; Xtt])
    # """  9000000   100000 """
    @NLexpression(model, PF, sum(200000 /(((Xtt[4i-3]-xa[1])^2+(Xtt[4i-2]-xa[2])^2)+0.01) for i=k_a+k_esc_cal-k_it:N-1))
    @constraint(model, conXt[i = 1:N-1], Xt[4i+1:4i+4] .== A* Xt[4i-3:4i] +B*U[2i-1:2i])
    @constraint(model, ucon[i=1:N], U[2i-1]^2 + U[2i]^2 <= u_max^2 )
    @constraint(model, vcon[i=1:N], Xt[4i-1]^2 + Xt[4i]^2 <= v_max^2)

    @NLobjective(model, Min, sum((Xtt[i]-X_goal[i])^2  for i=1:10) + PF )

    optimize!(model)
    U_solution = JuMP.value.(U)

    u_k = U_solution[1:2, :]

    return u_k
end
