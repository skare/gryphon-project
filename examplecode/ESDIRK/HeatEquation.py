# Copyright (C) 2012-2019 - Knut Erik Skare
#
# This file shows an example of the usage of Gryphon.
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

from dolfin import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction, Constant, Expression, inner, grad, dx, \
  Function, DOLFIN_EPS, DirichletBC
from gryphon import ESDIRK

# Define spatial mesh, function space, trial/test functions
mesh = UnitSquareMesh(29, 29)
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Define diffusion coefficient and source inside domain
D = Constant(0.1)
domainSource = Expression("10*sin(pi/2*t)*exp(-((x[0]-0.7)*(x[0]-0.7) + (x[1]-0.5)*(x[1]-0.5))/0.01)", t=0, degree=1)

# Define right hand side of the problem
rhs = -D * inner(grad(u), grad(v)) * dx + domainSource * v * dx

# Definie initial condition
W = Function(V)
W.interpolate(Constant(0.0))


# Define left and right boundary
def boundaryLeft(x, on_boundary):
    return x[0] < DOLFIN_EPS


def boundaryRight(x, on_boundary):
    return 1.0 - x[0] < DOLFIN_EPS


boundarySource = Expression("t", t=0, degree=1)
bcLeft = DirichletBC(V, boundarySource, boundaryLeft)
bcRight = DirichletBC(V, 0.0, boundaryRight)

# Define the time domain.
T = [0, 1]

# Create the ESDIRK object.
obj = ESDIRK(T, W, rhs, bcs=[bcLeft, bcRight], tdfBC=[boundarySource], tdf=[domainSource])

# Turn on extra terminal output
obj.parameters["verbose"] = True

# Turn on runtime plot of current time step
obj.parameters["drawplot"] = True

# Save runtime statistics.
obj.parameters["output"]["statistics"] = False

# Save plot of each time step in VTK format.
obj.parameters["output"]["plot"] = False

# Set that the plot of selected step sizes should be saved in jpg.
# Available choices are jpg, png and eps.
obj.parameters["output"]["imgformat"] = "jpg"

# Call the solver which will do the actual calculation.
obj.solve()
