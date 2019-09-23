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

from gryphon import ESDIRK
from dolfin import *

# Define spatial mesh, function space, trial/test functions
# TODO: Insert ref to mesh source.
mesh = Mesh("meshes/dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "meshes/dolfin_fine_subdomains.xml.gz")

V = FunctionSpace(mesh,"CG",1)
u = TrialFunction(V)
v = TestFunction(V)

# Define diffusion coefficient and source inside domain
D = Expression('sin(t*pi)',t=0)
bc0 = DirichletBC(V, D, sub_domains, 0)

# Define right hand side of the problem
rhs = -Constant(0.1)*inner(grad(u),grad(v))*dx

# Definie initial condition
W = Function(V)
W.interpolate(Constant(0.0))

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression('-t',t=0)
bc1 = DirichletBC(V, inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
# x0 = 0
zero = Expression('t',t=0)
bc2 = DirichletBC(V, zero, sub_domains, 2)

# Define the time domain.
T = [0.0,5.0]

# Create the ESDIRK object.
obj = ESDIRK(T,W,rhs,bcs=[bc0, bc1, bc2],tdfBC=[D,inflow,zero])

# Turn on extra terminal output
obj.parameters["verbose"] = True

# Turn on runtime plot of current time step
obj.parameters["drawplot"] = True

# Save runtime statistics.
obj.parameters["output"]["statistics"] = False

# Set tolerance for the adaptive timestepping
obj.parameters["timestepping"]["absolute_tolerance"] = 1e-4

# Call the solver which will do the actual calculation.
obj.solve()
