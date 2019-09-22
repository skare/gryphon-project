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
#
# This example shows how the heat equation stated as
# u_t = D*nabla^2(u) + f(u,t),
# where D is a constant and f is some function, can
# be solved using the Backward Euler method found in Gryphon.
#
# The example demonstrates how to handle explicit time dependencies on
# both the boundary and inside the domain.
#
# The problem will be solved on the unit square in the time interval
# T = [0,5].

from gryphon import backwardEuler
from dolfin import *

# Define spatial mesh, function space, trial/test functions
mesh = UnitSquareMesh(99,99)
V = FunctionSpace(mesh,"Lagrange",1)
u = TrialFunction(V)
v = TestFunction(V)

# Define diffusion coefficient and source inside the domain
D = Constant(0.1)
domainSource = Expression("10*sin(pi/2*t)*exp(-((x[0]-0.7)*(x[0]-0.7) + (x[1]-0.5)*(x[1]-0.5))/0.01)",t=0)

# Define right hand side of the problem (weak formulation)
rhs = -D*inner(grad(u),grad(v))*dx + domainSource*v*dx

# Define initial condition
W = Function(V)
W.interpolate(Constant(0.0))

# Define left and right boundary
def boundaryLeft(x,on_boundary):
  return x[0] < DOLFIN_EPS

def boundaryRight(x,on_boundary):
  return 1.0 - x[0] < DOLFIN_EPS

# Left boundary has an explicit time dependency
boundarySource = Expression("t/5",t=0)
bcLeft  = DirichletBC(V,boundarySource,boundaryLeft)
bcRight = DirichletBC(V,0.0,boundaryRight)

# Define the time domain
T = [0,0.5]

# Create the ImplicitEuler object
obj = backwardEuler(T,W,rhs,bcs=[bcLeft,bcRight],tdfBC=[boundarySource],tdf=[domainSource])

# Turn on some output and save run time
# statistics to sub folder "HeatEquation"
obj.parameters["verbose"] = True
obj.parameters["drawplot"] = True
obj.parameters["output"]["path"] = "HeatEquation"
obj.parameters["output"]["statistics"] = False
obj.parameters["timestepping"]["dt"] = 0.01

# Suppress some FEniCS output
set_log_level(WARNING)

# Solve the problem
obj.solve()

