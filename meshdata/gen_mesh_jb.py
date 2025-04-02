# Generates the journal-bearing problem mesh, or eccentric
#   rotating cylinders geometry with Gmsh for use in Fenics.
#   Actually accepts a meshsize parameter h instead of 
#   whatever meshsize/mesh density parameter Fenics uses lol

import gmsh
import sys
import meshio
from fenics import * 
import matplotlib.pyplot as plt


def main(h, inner_radius, ecc):
    gmsh.initialize() # must be done first

    # Create new gmsh model
    filename = "journal_bearing_h_%.4e"%h
    filepath = "meshdata/" + filename
    
    gmsh.model.add(filename)

    # shortcut. Didn't know Python could do this lol
    factory = gmsh.model.geo
    
    outer_radius = 1.0
    
    outer_center = (0.0, 0.0)
    inner_center = (0.0, -ecc)

    # addPoint() takes in x,y,z coordinates and a tag. If tag<0, sets automatically
    outer_center_pt = factory.addPoint(outer_center[0], outer_center[1], 0.0) 
    inner_center_pt = factory.addPoint(inner_center[0], inner_center[1], 0.0)

    # Mark NESW of inner and outer circles
    outer_n_pt = factory.addPoint(outer_center[0], outer_center[1]+outer_radius, 0.0, h) # North-most point
    outer_e_pt = factory.addPoint(outer_center[0]+outer_radius, outer_center[1], 0.0, h) # East-most point
    outer_s_pt = factory.addPoint(outer_center[0], outer_center[1]-outer_radius, 0.0, h)
    outer_w_pt = factory.addPoint(outer_center[0]-outer_radius, outer_center[1], 0.0, h)
    outer_points = [outer_n_pt, outer_e_pt, outer_s_pt, outer_w_pt]

    inner_n_pt = factory.addPoint(inner_center[0], inner_center[1]+inner_radius, 0.0, h)
    inner_e_pt = factory.addPoint(inner_center[0]+inner_radius, inner_center[1], 0.0, h)
    inner_s_pt = factory.addPoint(inner_center[0], inner_center[1]-inner_radius, 0.0, h) 
    inner_w_pt = factory.addPoint(inner_center[0]-inner_radius, inner_center[1], 0.0, h)
    inner_points = [inner_n_pt, inner_e_pt, inner_s_pt, inner_w_pt]


    # Add circle arcs. Must be less than pi, apparently, so we chop cirlce up into 4 arcs. Could have done 3 with trig
    outer_arcs = [0,0,0,0]
    inner_arcs = [0,0,0,0]
    for i in range(4):
        outer_arcs[i] =  factory.addCircleArc(outer_points[i], outer_center_pt, outer_points[(i+1)%4])
        inner_arcs[i] =  factory.addCircleArc(inner_points[i], inner_center_pt, inner_points[(i+1)%4])
            
    outer_loop = factory.addCurveLoop(outer_arcs)
    inner_loop = factory.addCurveLoop(inner_arcs)

    # outer_surface = factory.addPlaneSurface([outer_loop]) # is this needed?
    # Per the docs: The first curve loop defines the exterior contour; additional curve loop define holes.
    domain_surface = factory.addPlaneSurface([outer_loop, inner_loop])

    # Define boundaries (or any other marked parts of the mesh). For journal-bearing, there are 2
    outer_bndry_grp = gmsh.model.addPhysicalGroup(1, [outer_loop])
    gmsh.model.setPhysicalName(1, outer_bndry_grp, "outer_boundary")

    inner_bndry_grp = gmsh.model.addPhysicalGroup(1, [inner_loop])
    gmsh.model.setPhysicalName(1, inner_bndry_grp, "inner_boundary")

    domain_grp = gmsh.model.addPhysicalGroup(2, [domain_surface])
    gmsh.model.setPhysicalName(2, domain_grp, "Domain")
    
    # Synchronize the CAD (.geo) entities with the model
    gmsh.model.geo.synchronize()

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)

    # ... and save it to disk
    gmsh.write(filepath + ".msh")

    # Visualize mesh
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
        """"""

    # Always run this at the end
    gmsh.finalize()
    
    ##########################################################
    ####    Gmsh construction is over, now on to Fenics   ####
    ##########################################################
    
    
    # This function was recommended by Dokken in:
    # https://jsdokken.com/src/pygmsh_tutorial.html, "Mesh generation and conversion with GMSH and PYGMSH"
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        
        # Prune to 2D mesh if requested, ie ignore z component. mesh.prune_z_0() doesn't want to work
        points = mesh.points[:, :2] if prune_z else mesh.points
        
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
        
        return out_mesh
    
    # Read the Gmsh mesh
    msh = meshio.read(filepath + ".msh")

    # Create 2D mesh. "True" flag since it's a 2D mesh
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    meshio.write(filepath + "_triangle.xdmf", triangle_mesh)

    # Create 1D mesh
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write(filepath + "_line.xdmf", line_mesh)

    #print(".xdmf files written successfully")
    
    # If you had had a 3D mesh, you would need a 3D mesh and a 2D mesh 
    #   - No 1D mesh needed for a 3D mesh, I don't think
    # Replace 'triangle' with 'tetra' and 'line' with 'triangle'. Do not prune

    # Bring it back into FEniCS
    mymesh = Mesh()

    # 2D triangles 
    with XDMFFile(filepath + "_triangle.xdmf") as infile:
        infile.read(mymesh)
    mvc_2d = MeshValueCollection("size_t", mymesh, 2) 

    with XDMFFile(filepath + "_triangle.xdmf") as infile:
        infile.read(mvc_2d, "name_to_read")
    mf_2d = cpp.mesh.MeshFunctionSizet(mymesh, mvc_2d)

    # 1D lines
    mvc_1d = MeshValueCollection("size_t", mymesh, 1)

    with XDMFFile(filepath + "_line.xdmf") as infile:
        infile.read(mvc_1d, "name_to_read")
    mf_1d = cpp.mesh.MeshFunctionSizet(mymesh, mvc_1d)

    # Save mesh as .h5 file for easy FEniCS access, filepath.h5/mesh
    outfile = HDF5File(MPI.comm_world, filepath + ".h5", 'w')
    outfile.write(mymesh, '/mesh')
    outfile.close()
    
    """
    # Run this to recover the mesh from .h5 file for use in FEniCS:
    mesh2 = Mesh() #empty mesh
    infile = HDF5File(MPI.comm_world, filename + ".h5", 'r')
    infile.read(mesh2, '/mesh', True) #for some reason, need this flag to import a mesh?
    infile.close()
    print("mesh recovered from .h5 file, numel = %d"%mesh2.num_cells())
    plot(mesh2, title = "mesh recovered from .h5 file, numel = %d"%mesh2.num_cells())
    plt.show()
    
    # make sure boundary is detected properly
    P_elem = FiniteElement("CG", triangle, 1)
    P = FunctionSpace(mesh2, P_elem)
    
    rad = inner_radius
    # Boundaries of domain. The issue with triangles is that they always undershoot circles, at least here
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
    
    
    inner_bc = DirichletBC(P, Constant(1.0), Inner())
    outer_bc = DirichletBC(P, Constant(-1.0),  Outer())
    
    test_func = Function(P)
    inner_bc.apply(test_func.vector())
    outer_bc.apply(test_func.vector())
    
    fig = plot(test_func, title = "testing bc")
    plt.colorbar(fig)
    plt.show()
    """
    
    

# In case it's called from command line
if __name__ == '__main__':
    main(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    



