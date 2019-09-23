# Copyright (C) 2012-2019 - Knut Erik Skare
#
# This file shows an example of the usage of Gryphon.
# This example was inspired by the FEniCS example on periodic boundary
# conditions, which can be found by following the link below:
# http://fenicsproject.org/documentation/dolfin/1.4.0/python/demo/documented/periodic/python/documentation.html
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

from dolfin import *
from gryphon import ESDIRK


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) and on_boundary)


# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


# Create periodic boundary condition
pbc = PeriodicBoundary()

# Define spatial mesh, function space, trial/test functions
mesh = UnitSquareMesh(39, 39)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

u = TrialFunction(V)
v = TestFunction(V)

# Define diffusion coefficient and source inside domain
D = Constant(0.1)
domainSource = Expression("10*sin(pi/2*t)*exp(-((x[0]-1.0)*(x[0]-1.0) + (x[1]-0.5)*(x[1]-0.5))/0.01)", t=0, degree=1)

# Define right hand side of the problem
rhs = -D * inner(grad(u), grad(v)) * dx + domainSource * v * dx

# Definie initial condition
W = Function(V)
W.interpolate(Constant(0.0))

boundarySource = Expression("0.0", degree=1)
bcDirichlet = DirichletBC(V, boundarySource, DirichletBoundary())

# Define the time domain
T = [0, 1]

# Create the ESDIRK object
obj = ESDIRK(T, W, rhs, bcs=[bcDirichlet], tdf=[domainSource])

# Turn on extra terminal output
obj.parameters["verbose"] = True

# Turn on runtime plot of current time step
obj.parameters["drawplot"] = True

# Save runtime statistics.
obj.parameters["output"]["statistics"] = True

# Specify path that the output data will be stored in.
# This path can either be relative to where the script is being run
# from, or it can be absolute. To specify an absolute path, start
# the character '/' like this:
# obj.parameters["output"]["path"] = "/tmp/Heat_Equation_Absolute_Path"
# Here we have set up a relative path.
obj.parameters["output"]["path"] = "Heat_Equation_Relative_Path"

# If the following line is uncommented, the script will reuse the
# path if the simulation is repeated. If the line is kept commented,
# you will be asked whether or not the path can be reused. Note that
# if the path is reused, some files might be overwritten (plot files,
# statistics files etc.).
obj.parameters["output"]["reuseoutputfolder"] = True

# Call the solver which will do the actual calculation.
obj.solve()
