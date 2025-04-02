""" 
    Solves the steady-state incompressible Navier Stokes Equations
    for a few few common fluid flow problems using Taylor-Hood
    elements/a mixed formulation
"""

from fenics import *
from meshdata import gen_mesh_jb
import os
import math as math
import matplotlib.pyplot as plt

class Results: 
    def __init__(self, converged, velocity, pressure, stress_tensor, iters):
        self.converged = converged
        self.velocity = velocity
        self.pressure = pressure
        self.iters = iters


def navier_stokes_JB(h, rad, ecc, s, eta):
    # s is the tangential speed of the bearing 
    #print("NSE JB Solver called with h=%.5f"%h)

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
    f = Constant((0.0, 0.0)) # no body forces
    
    # Element spaces
    P_elem = FiniteElement("CG", triangle, 1) #pressure and auxiliary pressure, degree 1 elements
    V_elem = VectorElement("CG", triangle, 2) #velocity, degree 2 elements
    
    W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

    W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
    
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
    
    # Variational Problem: Trial and Test Functions
    w = TrialFunction(W) # nonlinear in w
    (u,pi) = split(w)
    (v, q) = TestFunctions(W)
    
    # eta*<del(u), del(v) > + <del(u).u, v> - <pi, div(v)> + <q, div(u)> 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx
    f_nse = inner(f,v)*dx
    
    F = a_nse - f_nse
    
    you = Function(W)
    F_act = action(F, you) 
    dF = derivative(F_act, you)

    problem = NonlinearVariationalProblem(F_act, you, bcs, dF)
    solver = NonlinearVariationalSolver(problem)
    try: 
        (iters, converged) = solver.solve()
    except: 
        print("Newton Method for Navier-Stokes failed to converge")
        return Results(False, None, None, None, 0)
    
    # Likewise, not sure which is preferred
    u_soln, p_soln = you.split(deepcopy=True)
    
    return Results(converged, u_soln, p_soln, iters)
 

def navier_stokes_LDC(h, s, eta):
    nx = round(1/h)
    mesh = UnitSquareMesh(nx, nx)
    print("Mesh loaded into FEniCS")

    # boundary data
    g_top = Expression(("s*16.0*x[0]*x[0]*(1-x[0])*(1-x[0])", "0.0"), s=s, degree = 4)
    g_walls = Constant((0.0, 0.0)) #g=0 on walls

    # body forces
    f = Constant((0.0, 0.0)) # no body forces
    
    # Element spaces
    P_elem = FiniteElement("CG", triangle, 1) #pressure and auxiliary pressure, degree 1 elements
    V_elem = VectorElement("CG", triangle, 2) #velocity, degree 2 elements

    W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

    W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
    
    # Interpolate body force and BCs onto velocity FE space
    g_top = interpolate(g_top, W.sub(0).collapse())
    g_walls = interpolate(g_walls, W.sub(0).collapse())
    f = interpolate(f, W.sub(0).collapse())
    
    # Define boundary conditions
    lid    = 'near(x[1], 1.0) && on_boundary'
    walls  = '(near(x[1], 0.0) || near(x[0], 0.0) || near(x[0], 1.0)) && on_boundary'
    corner = 'near(x[0], 0.0) && near(x[1], 0.0)' # for pressure regulating
    
    bc_top   = DirichletBC(W.sub(0), g_top, lid) # driving lid
    bc_walls = DirichletBC(W.sub(0), g_walls, walls) # no slip
    bc_press = DirichletBC(W.sub(1), Constant(0.0), corner, 'pointwise') # pressure regulating
    
    # Gather boundary conditions (any others would go here, separated by a comma)
    bcs = [bc_top, bc_walls, bc_press] #possibly get rid of pressure regulator if it breaks something in NN solve
    
    # Variational Problem: Trial and Test Functions
    w = TrialFunction(W) # nonlinear in w
    (u,pi) = split(w)
    (v, q) = TestFunctions(W)
    
    # eta*<del(u), del(v) > + <del(u).u, v> - <pi, div(v)> + <q, div(u)> 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx
    f_nse = inner(f,v)*dx
    
    F = a_nse - f_nse
    
    you = Function(W)
    F_act = action(F, you) 
    dF = derivative(F_act, you)

    problem = NonlinearVariationalProblem(F_act, you, bcs, dF)
    solver = NonlinearVariationalSolver(problem)
    try: 
        (iters, converged) = solver.solve()
    except: 
        print("Newton Method for Navier-Stokes failed to converge")
        return Results(False, None, None, None, 0)
    
    # Likewise, not sure which is preferred
    u_soln, p_soln = you.split(deepcopy=True)
        
    return Results(converged, u_soln, p_soln, iters)


def navier_stokes_LDC3D(h, s, eta):
    nx = round(1/h)
    mesh = UnitCubeMesh(nx, nx, nx)
    print("Mesh loaded into FEniCS")

    # boundary data
    g_top = Expression(("s*256.0*x[0]*x[0]*x[1]*x[1]*(1-x[0])*(1-x[0])*(1-x[1])*(1-x[1])", "0.0", "0.0"), s=s, degree = 4) 
    g_walls = Constant((0.0, 0.0, 0.0)) #g=0 on walls
    
    # body forces
    f = Constant((0.0, 0.0, 0.0)) # no body forces

    # Element spaces
    P_elem = FiniteElement("CG", tetrahedron, 1) #pressure and auxiliary pressure, degree 1 elements
    V_elem = VectorElement("CG", tetrahedron, 2) #velocity, degree 2 elements

    W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

    W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
    
    # Interpolate body force and BCs onto velocity FE space
    g_top = interpolate(g_top, W.sub(0).collapse())
    g_walls = interpolate(g_walls, W.sub(0).collapse())
    f = interpolate(f, W.sub(0).collapse())
    
    # Define boundary conditions
    top_lid = 'near(x[2], 1.0) && on_boundary'
    walls = '(near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0) || near(x[1], 1.0) || near(x[2], 0.0)) && on_boundary'
    origin = 'near(x[0], 0.0) && near(x[1], 0.0) && near(x[2], 0.0)' #for pressure regulating

    bc_top   = DirichletBC(W.sub(0), g_top, top_lid) # driving lid
    bc_walls = DirichletBC(W.sub(0), g_walls, walls) # no slip
    bc_press = DirichletBC(W.sub(1), Constant(0.0), origin, 'pointwise') # pressure regulating

    # Gather boundary conditions (any others would go here, separated by a comma)
    bcs = [bc_top, bc_walls, bc_press] #possibly get rid of pressure regulator if it breaks something in NN solve
    
    # Variational Problem: Trial and Test Functions
    w = TrialFunction(W) # nonlinear in w
    (u,pi) = split(w)
    (v, q) = TestFunctions(W)
    
    # eta*<del(u), del(v) > + <del(u).u, v> - <pi, div(v)> + <q, div(u)> 
    a_nse = eta*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx
    f_nse = inner(f,v)*dx
    
    F = a_nse - f_nse
    
    you = Function(W)
    F_act = action(F, you) 
    dF = derivative(F_act, you)

    problem = NonlinearVariationalProblem(F_act, you, bcs, dF)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = "newton"
    prm["newton_solver"]["linear_solver"] = "mumps" # utilizes parallel processors
    try: 
        (iters, converged) = solver.solve()
    except: 
        print("Newton Method for Navier-Stokes failed to converge")
        return Results(False, None, None, None, 0)
    
    # Likewise, not sure which is preferred
    u_soln, p_soln = you.split(deepcopy=True)
    
    return Results(converged, u_soln, p_soln, iters)


# Post-processing/new function stuff here
    
  
