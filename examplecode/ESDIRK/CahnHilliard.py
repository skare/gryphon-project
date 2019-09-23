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
import random

# Initial conditions
class InitialConditions(Expression):
  def __init__(self):
      random.seed(2 + MPI.process_number())
  def eval(self, values, x):
      values[0] = 0.63 + 0.02*(0.5 - random.random())
      values[1] = 0.0
  def value_shape(self):
      return (2,)

# Create mesh and define function spaces
mesh = UnitSquareMesh(49, 49)
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V*V

q,v = TestFunctions(ME)

# Define and interpolate initial condition
u   = Function(ME)
u.interpolate(InitialConditions())

c,mu = split(u)
c = variable(c)
f    = 100*c**2*(1-c)**2
dfdc = diff(f, c)
lmbda  = Constant(1.0e-02)

# Weak statement of the equations
f = -inner(grad(mu), grad(q))*dx
g = mu*v*dx - dfdc*v*dx - lmbda*inner(grad(c), grad(v))*dx

T = [0,5e-5] # Time domain

myobj = ESDIRK(T,u,f,g=g)
myobj.parameters['timestepping']['absolute_tolerance'] = 1e-2
myobj.parameters['timestepping']['inconsistent_initialdata'] = True
myobj.parameters['verbose'] = True
myobj.parameters['drawplot'] = True
myobj.parameters['output']['statistics'] = True

# Suppress some FEniCS output
set_log_level(WARNING)

# Solve the problem
myobj.solve()
