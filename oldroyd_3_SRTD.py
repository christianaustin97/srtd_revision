"""
    Christian Austin, University of Florida
    Part of Research for PhD Thesis, Summer of 2024
    
    Implements the SRTD formulation and recommended iterative 
    algorithm designed by Scott and Girault (2021) for the
    steady flow of a non-Newtonian fluid governed by a certain 
    3-parameter subset of the Oldroyd 8-parameter model 
    (Oldroyd, 1958)
    
    This program utilizes a variational approach and legacy
    (2019) FEniCS. The algorithm is iterative, with each 
    iteration containing 3-stages: the first stage involves 
    solving a Navier-Stokes-like equation for u and the 
    auxiliary pressure pi, then a linear transport equation 
    for the true pressure, and finally a linear transport 
    equation for the stress tensor. 
    
    This file contains built-in functions for solving the 
    lid-driven cavity (ldc) problem and the journal-bearing (jb)
    problem, as the analysis of the method assumes tangential
    Dirichlet boundary conditions. Hopefully more geometries to come soon. 
"""

from fenics import *
from meshdata import gen_mesh_jb
import os

class Results:
    def __init__(self, converged, velocity, aux_pressure, pressure, stress_tensor, residuals, Newton_iters):
        self.converged = converged
        self.velocity = velocity
        self.aux_pressure = aux_pressure
        self.pressure = pressure
        self.stress_tensor = stress_tensor
        self.residuals = residuals
        self.Newton_iters = Newton_iters


# Journal Bearing Problem

def oldroyd_3_JB_SRTD(h, rad, ecc, s, eta, l1, mu1, max_iter, tol):
    # s is the tangential speed of the bearing 

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

    # previous and next iterations. Symbolic when they are used in the weak forms, or pointers to the actual function values 
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
    T_return = as_tensor([[T_return_vec[0], T_return_vec[1]], [T_return_vec[1], T_return_vec[2]]])

    #LHS of NS-like solve, a((u,pi), (v,q)) 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx

    # RHS of NS-like stage is given in section 7 of Girault/Scott paper F((v,q); u0, T0)
    term1 = inner(f, v - l1*dot(grad(v), u0))*dx #orange term
    term2 = (p0*inner(nabla_grad(u0), grad(v)))*dx  #blue term
    term3 = -inner( dot(grad(u0),u0) , dot(grad(v),u0) )*dx #red term
    term4 = inner( dot(grad(u0),T0) , grad(v) )*dx #light green term
    term5 = inner( dot(sym(grad(u0)),T0)+dot(T0,sym(grad(u0))) , grad(v) )*dx #dark green term
    
    L_nse = term1 - l1*(term2 + term3 + term4) + (l1-mu1)*term5 #mathcal F 
    
    # Nonlinear in u, so must solve a-L==0 and use Newton instead of a==L directly
    F = a_nse - L_nse

    # Nonlinear NSE, so using Newton iteration
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
    while(n<=max_iter and min_residual > tol):
        try: 
            (Newton_iters[n], converged) = nse_solver.solve() # updates w1
        except: 
            print("Newton Method in the Navier-Stokes-like stage failed to converge")
            return Results(False, u_return, pi_return, p_return, T_return, residuals, Newton_iters)
        
        u_next, pi_next = w1.split(deepcopy=True)
        assign(u1, u_next) # u1 updated
        assign(pi1, pi_next) # pi1 updated

        p_solver.solve() # p1 updated

        T_solver.solve() # T1_vec updated
        T1 = as_tensor([[T1_vec[0], T1_vec[1]], [T1_vec[1], T1_vec[2]]]) # reshape to appropriate 

        # End of this SRTD iteration
        l2diff = errornorm(u1, u0, norm_type='l2', degree_rise=0)
        residuals[n] = l2diff
        if(l2diff <= min_residual):
            min_residual = l2diff
            u_return = u1
            pi_return = pi1
            p_return = p1
            T_return = T1

        print("SRTD Iteration %d: r = %.4e (tol = %.3e)" % (n, l2diff, tol))
        n = n+1
        
        #update u0, p0, T0
        assign(u0, u1)
        assign(p0, p1)
        assign(T0_vec, T1_vec) # can't assign T1 to T0 as a tensor, unfortunately
        
    # Stuff to do after the iterations are over
    if(min_residual <= tol):
        converged = True
    else:
        converged = False
    return Results(converged, u_return, pi_return, p_return, T_return, residuals, Newton_iters)


# Lid-Driven Cavity Problem

def oldroyd_3_LDC_SRTD(h, s, eta, l1, mu1, max_iter, tol):
    # s is the average velocity of the top lid

    nx = round(1/h)
    mesh = UnitSquareMesh(nx, nx)
    print("Mesh loaded into FEniCS")

    # boundary data
    g_top = Expression(("s*16.0*x[0]*x[0]*(1-x[0])*(1-x[0])", "0.0"), s=s, degree = 4) # 30x^2(1-x)^2, 30 gives it integral=1
    #g_top = Constant((float(s), 0.0))
    g_walls = Constant((0.0, 0.0)) #g=0 on walls

    # body forces
    f = Constant((0.0, 0.0)) # no body forces
    
    # Element spaces
    P_elem = FiniteElement("CG", triangle, 1) #pressure and auxiliary pressure, degree 1 elements
    V_elem = VectorElement("CG", triangle, 2) #velocity, degree 2 elements
    T_elem = VectorElement("CG", triangle, 2, dim=3) #stress tensor, degree 2 elements 
    
    W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

    W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
    P = FunctionSpace(mesh, P_elem) # true pressure space
    V = FunctionSpace(mesh, V_elem) # velocity space (not used)
    T = FunctionSpace(mesh, T_elem) # tensor space
    
    # Interpolate body force and BCs onto velocity FE space
    g_top = interpolate(g_top, W.sub(0).collapse())
    g_walls = interpolate(g_walls, W.sub(0).collapse())
    f = interpolate(f, W.sub(0).collapse())
    
    # Define boundary conditions
    top_lid = 'near(x[1], 1.0) && on_boundary'
    walls = '(near(x[1], 0.0) || near(x[0], 0.0) || near(x[0], 1.0)) && on_boundary'
    origin = 'near(x[0], 0.0) && near(x[1], 0.0)' #for pressure regulating

    bc_top = DirichletBC(W.sub(0), g_top, top_lid)
    bc_walls = DirichletBC(W.sub(0), g_walls, walls)
    pressure_reg = DirichletBC(W.sub(1), Constant(0.0), origin, 'pointwise')
        
    # Gather boundary conditions (any others would go here, separated by a comma)
    bcs = [bc_top, bc_walls, pressure_reg] 
    
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

    # previous and next iterations. Symbolic when they are used in the weak forms, or pointers to the actual function values 
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

    # Functions we'll actually return
    u_return = Function(V)
    pi_return = Function(P)
    p_return = Function(P)
    T_return_vec = Function(T)
    T_return = as_tensor([[T_return_vec[0], T_return_vec[1]], [T_return_vec[1], T_return_vec[2]]])


    #LHS of NS-like solve, a((u,pi), (v,q)) 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx

    # RHS of NS-like stage is given in section 7 of Girault/Scott paper F((v,q); u0, T0)
    term1 = inner(f, v - l1*dot(grad(v), u0))*dx #orange term
    term2 = (p0*inner(nabla_grad(u0), grad(v)))*dx  #blue term
    term3 = -inner( dot(grad(u0),u0) , dot(grad(v),u0) )*dx #red term
    term4 = inner( dot(grad(u0),T0) , grad(v) )*dx #light green term
    term5 = inner( dot(sym(grad(u0)),T0)+dot(T0,sym(grad(u0))) , grad(v) )*dx #dark green term
    
    L_nse = term1 - l1*(term2 + term3 + term4) + (l1-mu1)*term5 #mathcal F 
    
    # Nonlinear in u, so must solve a-L==0 and use Newton instead of a==L directly
    F = a_nse - L_nse

    # Nonlinear NSE, so using Newton iteration
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
    while(n<=max_iter and min_residual > tol):
        try: 
            (Newton_iters[n], converged) = nse_solver.solve() # updates w1
        except: 
            print("Newton Method in the Navier-Stokes-like stage failed to converge")
            return Results(False, u_return, pi_return, p_return, T_return, residuals, Newton_iters)
        
        u_next, pi_next = w1.split(deepcopy=True)
        assign(u1, u_next) # u1 updated
        assign(pi1, pi_next) # pi1 updated

        p_solver.solve() # p1 updated

        T_solver.solve() # T1 updated
        T1 = as_tensor([[T1_vec[0], T1_vec[1]], [T1_vec[1], T1_vec[2]]]) # reshape to appropriate 

        # End of this SRTD iteration
        l2diff = errornorm(u1, u0, norm_type='l2', degree_rise=0)
        residuals[n] = l2diff
        if(l2diff <= min_residual):
            min_residual = l2diff
            u_return = u1
            pi_return = pi1
            p_return = p1
            T_return = T1

        print("SRTD Iteration %d: r = %.4e (tol = %.3e)" % (n, l2diff, tol))
        n = n+1
        
        #update u0, p0, T0
        assign(u0, u1)
        assign(p0, p1)
        assign(T0_vec, T1_vec)    
        
    # Stuff to do after the iterations are over
    if(min_residual <= tol):
        converged = True
    else:
        converged = False
    return Results(converged, u_return, pi_return, p_return, T_return, residuals, Newton_iters)

     
def oldroyd_3_LDC3D_SRTD(h, s, eta, l1, mu1, max_iter, tol):
    nx = round(1/h)
    mesh = UnitCubeMesh(nx, nx, nx)
    print("Mesh loaded into FEniCS")

    # boundary data
    g_top = Expression(("s*256.0*x[0]*x[0]*x[1]*x[1]*(1-x[0])*(1-x[0])*(1-x[1])*(1-x[1])", "0.0", "0.0"), s=s, degree = 4) 
    #g_top = Constant((float(s), 0.0))
    g_walls = Constant((0.0, 0.0, 0.0)) #g=0 on walls
    
    # body forces
    f = Constant((0.0, 0.0, 0.0)) # no body forces
    
    # Element spaces
    P_elem = FiniteElement("CG", tetrahedron, 1) #pressure and auxiliary pressure, degree 1 elements
    V_elem = VectorElement("CG", tetrahedron, 2) #velocity, degree 2 elements
    T_elem = VectorElement("CG", tetrahedron, 2, dim=6) #stress tensor, degree 2 elements 
    
    W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

    W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
    P = FunctionSpace(mesh, P_elem) # true pressure space
    V = FunctionSpace(mesh, V_elem) # velocity space 
    T = FunctionSpace(mesh, T_elem) # tensor space
    
    # Interpolate body force and BCs onto velocity FE space
    g_top = interpolate(g_top, W.sub(0).collapse())
    g_walls = interpolate(g_walls, W.sub(0).collapse())
    f = interpolate(f, W.sub(0).collapse())
    
    # Define boundary conditions
    top_lid = 'near(x[2], 1.0) && on_boundary'
    walls = '(near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0) || near(x[1], 1.0) || near(x[2], 0.0)) && on_boundary'
    origin = 'near(x[0], 0.0) && near(x[1], 0.0) && near(x[2], 0.0)' #for pressure regulating

    bc_top = DirichletBC(W.sub(0), g_top, top_lid)
    bc_walls = DirichletBC(W.sub(0), g_walls, walls)
    pressure_reg = DirichletBC(W.sub(1), Constant(0.0), origin, 'pointwise')
        
    # Gather boundary conditions (any others would go here, separated by a comma)
    bcs = [bc_top, bc_walls, pressure_reg] 
    
    # Variational Problem Begin
    #
    # Trial Functions. Think of TrialFunctions as symbolic, and they are only used in defining the weak forms
    w = TrialFunction(W) # our NS-like TrialFunction
    (u,pi) = split(w) # trial functions, representing u1, pi1
    p = TrialFunction(P) # true pressure trial function for auxiliary pressure equation, representing p1
    tau_vec = TrialFunction(T) # stress trial function for stress tensor equation, representing T1
    tau = as_tensor([[tau_vec[0], tau_vec[1], tau_vec[2]], \
                     [tau_vec[1], tau_vec[3], tau_vec[4]], \
                     [tau_vec[2], tau_vec[4], tau_vec[5]]])

    # Weak form test functions. Also think of these as symbolic, and they are only used in defining the weak forms
    (v, q) = TestFunctions(W) # test functions for NSE step
    r = TestFunction(P) # test functions for pressure transport
    S_vec = TestFunction(T) # test functions for constitutive equation
    S = as_tensor([[S_vec[0], S_vec[1], S_vec[2]], \
                   [S_vec[1], S_vec[3], S_vec[4]], \
                   [S_vec[2], S_vec[4], S_vec[5]]])

    # previous and next iterations. Symbolic when they are used in the weak forms, or pointers to the actual function values 
    #w0 = Function(W)
    u0 = Function(V)    
    #pi0 = Function(P)
    p0 = Function(P)
    T0_vec = Function(T)
    T0 = as_tensor([[T0_vec[0], T0_vec[1], T0_vec[2]], \
                    [T0_vec[1], T0_vec[3], T0_vec[4]], \
                    [T0_vec[2], T0_vec[4], T0_vec[5]]])

    w1 = Function(W)
    u1 = Function(V)
    pi1 = Function(P)
    p1 = Function(P)
    T1_vec = Function(T)

    # Functions we'll actually return
    u_return = Function(V)
    pi_return = Function(P)
    p_return = Function(P)
    T_return_vec = Function(T)
    T_return = as_tensor([[T_return_vec[0], T_return_vec[1], T_return_vec[2]], \
                          [T_return_vec[1], T_return_vec[3], T_return_vec[4]], \
                          [T_return_vec[2], T_return_vec[4], T_return_vec[5]]])


    #LHS of NS-like solve, a((u,pi), (v,q)) 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx

    # RHS of NS-like stage is given in section 7 of Girault/Scott paper F((v,q); u0, T0)
    term1 = inner(f, v - l1*dot(grad(v), u0))*dx #orange term
    term2 = (p0*inner(nabla_grad(u0), grad(v)))*dx  #blue term
    term3 = -inner( dot(grad(u0),u0) , dot(grad(v),u0) )*dx #red term
    term4 = inner( dot(grad(u0),T0) , grad(v) )*dx #light green term
    term5 = inner( dot(sym(grad(u0)),T0)+dot(T0,sym(grad(u0))) , grad(v) )*dx #dark green term
    
    L_nse = term1 - l1*(term2 + term3 + term4) + (l1-mu1)*term5 #mathcal F 
    
    # Nonlinear in u, so must solve a-L==0 and use Newton instead of a==L directly
    F = a_nse - L_nse

    # Nonlinear NSE, so using Newton iteration
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

    # Begin SRTD iteration
    n=1
    l2diff = 1.0
    residuals = {} # empty dict to save residual value after each iteration 
    Newton_iters = {}
    min_residual = 1.0
    while(n<=max_iter and min_residual > tol):

        try: 
            (Newton_iters[n], converged) = nse_solver.solve() # updates w1
        except: 
            print("Newton Method in the Navier-Stokes-like stage failed to converge")
            return Results(False, u_return, pi_return, p_return, T_return, residuals, Newton_iters)
        

        u_next, pi_next = w1.split(deepcopy=True) 
        assign(u1, u_next) # u1 updated
        assign(pi1, pi_next) # pi1 updated

        p_solver.solve() # p1 updated

        T_solver.solve() # T1_vec updated
        T1 = as_tensor([[T1_vec[0], T1_vec[1], T1_vec[2]], \
                        [T1_vec[1], T1_vec[3], T1_vec[4]], \
                        [T1_vec[2], T1_vec[4], T1_vec[5]]])

        # End of this SRTD iteration
        l2diff = errornorm(u1, u0, norm_type='l2', degree_rise=0)
        residuals[n] = l2diff
        if(l2diff <= min_residual):
            min_residual = l2diff
            u_return = u1
            pi_return = pi1
            p_return = p1
            T_return = T1
        
        print("SRTD Iteration %d: r = %.4e (tol = %.3e)" % (n, l2diff, tol))
        n = n+1

        #update u0, p0, T0
        assign(u0, u1)
        assign(p0, p1)
        assign(T0_vec, T1_vec)

    # Stuff to do after the iterations are over
    if(min_residual <= tol):
        converged = True
    else:
        converged = False
    return Results(converged, u_return, pi_return, p_return, T_return, residuals, Newton_iters)


#post proc stuff here




