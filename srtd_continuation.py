from fenics import *
from meshdata import gen_mesh_jb
import numpy as np
import os
import matplotlib.pyplot as plt

class Results:
    def __init__(self, lambda_max, velocity, aux_pressure, pressure, stress_tensor):
        self.lambda_max = lambda_max
        self.velocity = velocity
        self.aux_pressure = aux_pressure
        self.pressure = pressure
        self.stress_tensor = stress_tensor


def JB_SRTD_continuation(h, rad, ecc, s, eta, l_max, mu1, max_srtd_iters, tol, continuation_scheme, continuation_steps):
    if(rad>=1 or rad<=0 or ecc<0 or rad+ecc>1):
        #throw exception, forgot how lol
        print("Error: Inputs not valid")
    
    meshfile = "meshdata/journal_bearing_h_%.4e.h5"%h
    
    if not os.path.exists(meshfile):
        print("Creating mesh...")
        gen_mesh_jb.main(h, rad, ecc)
    
    #then, simply read the mesh in 
    mesh = Mesh() #empty mesh
    infile = HDF5File(MPI.comm_world, meshfile, 'r')
    infile.read(mesh, '/mesh', True) #for some reason, need this flag to import a mesh?
    infile.close()
    print("Mesh loaded into FEniCS")

    # everything up to defining the RHS and LHS is the same for any value of lambda
    
    #boundaries of domain
    class Inner(SubDomain):
        def inside(self, x, on_boundary):
            radius = x[0]*x[0] + (x[1]+ecc)*(x[1]+ecc)
            return (on_boundary and radius <= rad*rad+h)
    
    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            # on_boundary and opposite of inner lol
            radius = x[0]*x[0] + (x[1]+ecc)*(x[1]+ecc)
            return (on_boundary and radius > rad*rad+h)
    
    class TopPoint(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], 0.0) and near(x[1], 1.0))
    
    # Boundary data     
    speed_outer = 0.0 # counter-clockwise tangential speed of outer bearing, 
    speed_inner = s # clockwise tangential speed of inner bearing, 
    g_inner = Expression(("s*(x[1]+ecc)/r", "-s*x[0]/r"), s=speed_inner, r=rad, ecc=ecc, degree=1) 
    g_outer = Expression(("-s*x[1]", "s*x[0]"), s=speed_outer, degree=1) # no slip on outer bearing 
    
    # body forces
    f = Constant((0.0, 0.0)) # no body forces
    
    # Element spaces
    P_elem = FiniteElement("CG", triangle, 1) # Pressure and auxiliary pressure, degree 1 elements
    V_elem = VectorElement("CG", triangle, 2) # Velocity, degree 2 elements
    T_elem = VectorElement("CG", triangle, 2, dim=3) # Stress tensor has 3 independent components
    
    W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

    W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
    P = FunctionSpace(mesh, P_elem) # true pressure space
    V = FunctionSpace(mesh, V_elem) # velocity space 
    T = FunctionSpace(mesh, T_elem) # tensor space
    
    # Interpolate body force and BCs onto velocity FE space
    g_inner = interpolate(g_inner, W.sub(0).collapse())
    g_outer = interpolate(g_outer, W.sub(0).collapse())
    f = interpolate(f, W.sub(0).collapse())

    # Define boundary conditions
    bc_inner = DirichletBC(W.sub(0), g_inner, Inner())
    bc_outer = DirichletBC(W.sub(0), g_outer, Outer())
    bc_press = DirichletBC(W.sub(1), Constant(0.0), TopPoint(), 'pointwise')
    
    # Gather boundary conditions (any others would go here, separated by a comma)
    bcs = [bc_inner, bc_outer, bc_press] 
    
    # Variational Problem Begin
    #
    # Trial Functions. Think of TrialFunctions as symbolic, and they are only used in defining the weak forms
    w = TrialFunction(W) # our NS-like TrialFunction
    (u,pi) = split(w) # trial functions, representing u1, pi1
    p = TrialFunction(P) # true pressure trial function for auxiliary pressure equation, representing p1
    tau_vec = TrialFunction(T) # stress trial function for stress tensor equation, representing T1
    tau = as_tensor([[tau_vec[0], tau_vec[1]], [tau_vec[1], tau_vec[2]]])

    # Weak form test functions. Also think of these as symbolic, and they are only used in defining the weak forms
    (v, q) = TestFunctions(W) # test functions for NSE step
    r = TestFunction(P) # test functions for pressure transport
    S_vec = TestFunction(T) # test functions for constitutive equation
    S = as_tensor([[S_vec[0], S_vec[1]], [S_vec[1], S_vec[2]]])

    # previous and next SRTD iterations. Symbolic when they are used in the weak forms, or pointers to the actual function values 
    #w0 = Function(W)
    u0 = Function(V)    
    #pi0 = Function(P)
    p0 = Function(P)
    T0_vec = Function(T)
    T0 = as_tensor([[T0_vec[0], T0_vec[1]], [T0_vec[1], T0_vec[2]]]) 

    w1 = Function(W)
    u1 = Function(V)
    pi1 = Function(P)
    p1 = Function(P)
    T1_vec = Function(T)
    
    # Functions we'll actually return,
    u_return = Function(V)
    pi_return = Function(P)
    p_return = Function(P)
    T_return_vec = Function(T)
    #T_return = as_tensor([[T_return_vec[0], T_return_vec[1]], [T_return_vec[1], T_return_vec[2]]])

    #LHS of NS-like solve, a((u,pi), (v,q)) 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx

    # we can factor l1 out of many terms in the RHS given in section 7 of Girault/Scott paper F((v,q); u0, T0)
    term2 = (p0*inner(nabla_grad(u0), grad(v)))*dx  #blue term
    term3 = -inner( dot(grad(u0),u0) , dot(grad(v),u0) )*dx #red term
    term4 = inner( dot(grad(u0),T0) , grad(v) )*dx #light green term
    term5 = inner( dot(sym(grad(u0)),T0)+dot(T0,sym(grad(u0))) , grad(v) )*dx #dark green term

    # begin continuation method

    # solution values at previous lambda val
    u_lambda = Function(V)
    p_lambda = Function(P)
    T_lambda_vec = Function(T)
    # solution values at 2 previous lambda val
    u_lambda_m_dl = Function(V)
    p_lambda_m_dl = Function(P)
    T_lambda_m_dl_vec = Function(T)

    continuation_iters = 0

    fig = plt.figure()

    l1_vals = np.linspace(0, l_max, continuation_steps)
    for l1 in l1_vals:
        # solve SRTD for this given l1
        l1 = float(l1)
        mu1 = l1
        print("="*100)
        print("Solving Continuation step %d, l1 = %.4e\n"%(continuation_iters, l1))

        # First term of RHS of NS-like stage
        term1 = inner(f, v - l1*dot(grad(v), u0))*dx #orange term
        L_nse = term1 - l1*(term2 + term3 + term4) + (l1-mu1)*term5 #mathcal F 
        
        # Nonlinear in u, so must solve a-L==0 and use Newton instead of a==L directly
        F = a_nse - L_nse

        # Nonlinear NSE, so using Newton iteration
        # w1 should contain the initial Newton step 
        F_act = action(F, w1) 
        dF = derivative(F_act, w1)
        nse_problem = NonlinearVariationalProblem(F_act, w1, bcs, dF) # will update w1 values every time we call solver.solve()
        nse_solver = NonlinearVariationalSolver(nse_problem)
        nse_prm = nse_solver.parameters
        nse_prm["nonlinear_solver"] = "newton"
        nse_prm["newton_solver"]["linear_solver"] = "mumps" # utilizes parallel processors

        # Pressure transport equation
        ap = (p + l1*dot(grad(p), u1)) * r * dx 
        Lp = pi1 * r * dx 
        
        p1 = Function(P)
        p_problem = LinearVariationalProblem(ap, Lp, p1) # will update p1 values every time we call solver.solve()
        p_solver = LinearVariationalSolver(p_problem)

        
        # Stress transport equation/Constitutive equation
        aT = inner( tau + l1*(dot(grad(tau),u1) + dot(-skew(grad(u1)), tau) - dot(tau, -skew(grad(u1)))) \
                            - mu1*(dot(sym(grad(u1)), tau) + dot(tau, sym(grad(u1)))) , S)*dx
        LT = 2.0*eta*inner(sym(grad(u1)), S)*dx

        T_problem = LinearVariationalProblem(aT, LT, T1_vec) # will update T1_vec values every time we call solver.solve()
        T_solver = LinearVariationalSolver(T_problem)
        T_prm = T_solver.parameters
        T_prm["linear_solver"] = "mumps"

        # Begin SRTD iterative solve
        n=1
        l2diff = 1.0
        residuals = {} # empty dict to save residual value after each iteration
        Newton_iters = {}
        min_residual = 1.0
        while(n<=max_srtd_iters and min_residual > tol):
            try: 
                (Newton_iters[n], converged) = nse_solver.solve() # updates w1
            except: 
                print("Newton Method in the Navier-Stokes-like stage failed to converge")
                #return Results(False, u_return, pi_return, p_return, T_return, residuals, Newton_iters)
            
            u_next, pi_next = w1.split(deepcopy=True)
            assign(u1, u_next) # u1 updated
            assign(pi1, pi_next) # pi1 updated

            p_solver.solve() # p1 updated

            T_solver.solve() # T1_vec updated
            T1 = as_tensor([[T1_vec[0], T1_vec[1]], [T1_vec[1], T1_vec[2]]]) # reshape to appropriate 

            # End of this SRTD iteration
            if converged:
                l2diff = errornorm(u1, u0, norm_type='l2', degree_rise=0)
            else:
                l2diff = 1.0
            residuals[n] = l2diff
            if(l2diff <= min_residual):
                l1_return = l1 # means we hit some local minimum for this l1, it didn't immediately explode
                min_residual = l2diff
                u_return.assign(u1)
                pi_return.assign(pi1)
                p_return.assign(p1)
                T_return_vec.assign(T1_vec)

            print("SRTD Iteration %d completed: r = %.4e (tol = %.3e)" % (n, l2diff, tol))
            n = n+1
            
            #update u0, p0, T0
            assign(u0, u1)
            assign(p0, p1)
            assign(T0_vec, T1_vec) # can't assign T1 to T0 as a tensor, unfortunately
            
        # after SRTD converged for this l1, update counter ...
        continuation_iters += 1

        # determine continuation scheme
        if continuation_scheme == 0:
            print("No continuation, restarting SRTD iteration at 0...")
            u0.assign(Constant((0.0, 0.0)))
            p0.assign(Constant(0.0))
            T0_vec.assign(Constant((0.0, 0.0, 0.0)))

        elif continuation_scheme == 1:
            print("Using natural parameter continuation...")
            # best guess from last solve saved in var_return
            u0.assign(u_return)
            p0.assign(p_return)
            T0_vec.assign(T_return_vec)
        
        elif continuation_scheme == 2:
            print("Using secant line predictor continuation...")
            # update two-previous l1 solutions
            u_lambda_m_dl.assign(u_lambda)
            p_lambda_m_dl.assign(p_lambda)
            T_lambda_m_dl_vec.assign(T_lambda_vec)
            
            # best guess from last SRTD solve is stored in var_return
            u_lambda.assign(u_return)
            p_lambda.assign(p_return)
            T_lambda_vec.assign(T_return_vec)

            if(continuation_iters <= 1):
                # then use natural parameter predictor
                print("Not enough continuation iters, using last solution as starting guess/predictor")
                u0.assign(u_lambda)
                p0.assign(p_lambda)
                T0_vec.assign(T_lambda_vec)
            else:
                # we have at least two previous terms, so use the linear secant predictor
                print("Using secant line predictor")
                u0.assign(2*u_lambda - u_lambda_m_dl)
                p0.assign(2*p_lambda - p_lambda_m_dl)
                T0_vec.assign(2*T_lambda_vec - T_lambda_m_dl_vec)
        else:
            raise ValueError("Continuation scheme not defined!")

        T0 = as_tensor([[T0_vec[0], T0_vec[1]], [T0_vec[1], T0_vec[2]]]) # T0 updated
        plt.semilogy(residuals.keys(), residuals.values(), label='$\lambda_{1} = $%1.4e'%l1)
        
    
    # once continuation iteration is done, plot and return values
    plt.hlines(y=tol,xmin = 0, xmax=max_srtd_iters, linestyles = '--', label="Tol: %1.1e"%tol)
    if continuation_scheme == 0:
        plt.title("Residual vs iteration, no continuation")
    elif continuation_scheme == 1:
        plt.title("Residual vs iteration, natural paraemter continuation")
    elif continuation_scheme == 2:
        plt.title("Residual vs iteration, secant line predictor continuation")
    else:
        plt.title("If you see this, somethign is seriously broken")
    plt.legend() 
    plt.xlabel("SRTD iteration")
    plt.ylabel("residual")
    plt.show()
    return Results(l1_return, u_return, pi_return, p_return, T_return_vec)
