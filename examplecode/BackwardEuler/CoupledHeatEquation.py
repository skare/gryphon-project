# Copyright (C) 2012-2014 - Knut Erik Skare
#
# Gryphon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gryphon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gryphon. If not, see <http://www.gnu.org/licenses/>.


from gryphon import backwardEuler,ESDIRK
from dolfin import *
from numpy import power,pi,sin

class InitialConditions(Expression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Define mesh, function space and test functions
mesh = RectangleMesh(0.0, 0.0, 1.0, 1.0, 49, 49)
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V*V
q1,q2 = TestFunctions(ME)
U1,U2 = TrialFunctions(ME)

# Define and interpolate initial condition
W = Function(ME)
W.interpolate(InitialConditions())
u,v = split(W)

domainSource = Expression("0.1*t*exp(-((x[0]-0.7)*(x[0]-0.7) + (x[1]-0.5)*(x[1]-0.5))/0.01)",t=0)

# Define the right hand side for each of the PDEs
F1 = (-inner(grad(U1),grad(q1)) + U2*q1 + domainSource*q1)*dx
F2 = (-inner(grad(U2),grad(q2)) - U1*q2)*dx

# Define Dirichlet boundary, zero on all edges.
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

bc_u = DirichletBC(ME.sub(0), 0.0, boundary)
bc_v = DirichletBC(ME.sub(1), 0.0, boundary)

# Define the time domain
T = [0,10]

# Create the solver object and adjust tolerance
obj = backwardEuler(T,W,[F1,F2],tdf=[domainSource], bcs=[bc_u,bc_v])

# Turn on some output and save run time
# statistics to sub folder "GrayScott"
obj.parameters["verbose"] = True
obj.parameters["drawplot"] = True
obj.parameters["output"]["path"] = "CoupledHeatEquation_folder"
obj.parameters["output"]["statistics"] = True
obj.parameters["timestepping"]["dt"] = T[1]/40.0

# Suppress some FEniCS output
set_log_level(WARNING)

# Solve the problem
obj.solve()
