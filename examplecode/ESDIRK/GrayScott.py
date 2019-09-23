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
from numpy import power, pi, sin


class InitialConditions(UserExpression):
    def eval(self, values, x):
        if between(x[0], (0.75, 1.25)) and between(x[1], (0.75, 1.25)):
            values[1] = 0.25 * power(sin(4 * pi * x[0]), 2) * power(sin(4 * pi * x[1]), 2)
            values[0] = 1 - 2 * values[1]
        else:
            values[1] = 0
            values[0] = 1

    def value_shape(self):
        return (2,)


# Define mesh, function space and test functions
mesh = RectangleMesh(Point(0, 0), Point(2, 2), 49, 49, "right/left")
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1 * P1)
q1, q2 = TestFunctions(ME)

# Define and interpolate initial condition
W = Function(ME)
W.interpolate(InitialConditions())
u, v = split(W)

# Define parameters in Gray-Scott model
Du = Constant(8.0e-5)
Dv = Constant(4.0e-5)
F = Constant(0.024)
k = Constant(0.06)

# Define the right hand side for each of the PDEs
F1 = (-Du * inner(grad(u), grad(q1)) - u * (v ** 2) * q1 + F * (1 - u) * q1) * dx
F2 = (-Dv * inner(grad(v), grad(q2)) + u * (v ** 2) * q2 - (F + k) * v * q2) * dx

# Define the time domain
T = [0, 2500]

# Create the solver object and adjust tolerance
obj = ESDIRK(T, W, [F1, F2])
obj.parameters["timestepping"]["absolute_tolerance"] = 1e-3

# Turn on some output and save run time
# statistics to sub folder "GrayScott"
obj.parameters["verbose"] = True
obj.parameters["drawplot"] = False  # Drawplot seems to be pretty broken, do not set to True
obj.parameters["output"]["path"] = "GrayScott"
obj.parameters["output"]["statistics"] = True
obj.parameters["output"]["plot"] = True

# Supress some FEniCS output
set_log_level(LogLevel.WARNING)

# Solve the problem
obj.solve()
