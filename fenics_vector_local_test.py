from fenics import *
import matplotlib.pyplot as plt
import time
import numpy as np

mesh = UnitSquareMesh(20,20)
V_elem = FiniteElement("CG", triangle, 1)
V = FunctionSpace(mesh, V_elem)

u = Expression("x[0] + x[1]", degree = 1)
v = Expression("x[0]*x[1]", degree = 2)
u = interpolate(u, V)
v = interpolate(v, V)

figu = plot(u)
plt.colorbar(figu)
plt.title("$\mathbf{u}$")
plt.show()

figv = plot(v)
plt.colorbar(figv)
plt.title("$\mathbf{v}$")
plt.show()

# notice that we cannot assign a sum
x = Function(V) 
try:
    assign(x, u+v) # throws an error
except:
    print("error! Cannot assign a function a value of an algebraic sum")

# so 3 options: Function.assign(sending_func), assign(receiving_func, sending_func), or set_local and get_local
x1 = Function(V)
assign(x1, project(u+2*v, V))
figx1 = plot(x1)
plt.colorbar(figx1)
plt.title("sum using assign and project")
plt.show()

# or get local (this one is way faster)
x2 = Function(V)

times = np.zeros(100)
for i in range(100):
    start = time.perf_counter()
    x2.vector().set_local(u.vector().get_local() + 2*v.vector().get_local())
    end = time.perf_counter()
    times[i] = end-start # number of seconds as float
print("average time needed using set_local and get_local: %1.5es" %times.mean())

figx2 = plot(x2)
plt.colorbar(figx2)
plt.title("sum using local values")
plt.show()

# or I think regular function.assign() works, we'll find out
# or get local (this one is way faster)
x3 = Function(V)

times = np.zeros(100)
for i in range(100):
    start = time.perf_counter()
    x3.assign(u+2*v)
    end = time.perf_counter()
    times[i] = end-start # number of seconds as float
print("average time needed using u.assign(v): %1.5es"%times.mean())

figx3 = plot(x3)
plt.colorbar(figx3)
plt.title("sum using function.assign()")
plt.show()

# see differences
fig_dif1 = plot(abs(x1 - x3))
plt.colorbar(fig_dif1)
plt.title("|x1-x3|, x1 project, x3 assign")
plt.show()


# see differences
fig_dif2 = plot(abs(x2 - x3))
plt.colorbar(fig_dif2)
plt.title("|x2-x3|, x2 get_local, x3 assign")
plt.show()




