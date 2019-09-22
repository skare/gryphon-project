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
import numpy as np

mesh = UnitSquareMesh(29,29)
V = FunctionSpace(mesh,"Lagrange",1)
u = TrialFunction(V)
v = TestFunction(V)

D = Constant(0.1)
domainSource = Expression("10*sin(t)*exp(-((x[0]-0.7)*(x[0]-0.7) + (x[1]-0.5)*(x[1]-0.5))/0.01)",t=0)
rhs = -D*inner(grad(u),grad(v))*dx + domainSource*v*dx

# Define left and right boundary
def boundaryLeft(x,on_boundary):
  return x[0] < DOLFIN_EPS

def boundaryRight(x,on_boundary):
  return 1.0 - x[0] < DOLFIN_EPS

# Left boundary has an explicit time dependency
boundarySource = Expression("t",t=0)
bcLeft  = DirichletBC(V,boundarySource,boundaryLeft)
bcRight = DirichletBC(V,0.0,boundaryRight)

# Define the time domain
T = [0,1.0]

# Define initial condition
W_exact = Function(V)
W_exact.interpolate(Constant(0.0))

exactSolver = ESDIRK(T,W_exact,rhs,bcs=[bcLeft,bcRight],tdfBC=[boundarySource],tdf=[domainSource])
exactSolver.parameters['timestepping']['adaptive'] = False
exactSolver.parameters['verbose'] = True
exactSolver.parameters["timestepping"]["dt"] = 1e-4
exactSolver.solve()

timestepVector = [1e-4*np.power(2,i) for i in range(0,5)]
errorVector = []

for dt in timestepVector:
	W = Function(V)
	W.interpolate(Constant(0.0))

	# Create the ImplicitEuler object
	obj = backwardEuler(T,W,rhs,bcs=[bcLeft,bcRight],tdfBC=[boundarySource],tdf=[domainSource])
	obj.parameters["verbose"] = True
	obj.parameters["timestepping"]["dt"] = dt
	obj.solve()
	
	errorVector.append(norm(W_exact.vector() - W.vector(),'l2'))

result = np.polyfit(np.log(timestepVector),np.log(errorVector),1)
print "Resulting order coefficient: " + str(result[0]) + " should be close to 1.0 for Backward Euler."
