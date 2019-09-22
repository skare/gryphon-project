====================================================
Gryphon, a module for time integration within FEniCS
====================================================
Currently supports FEniCS 1.4.0 and 1.5.0

|
Gryphon is intended to serve as a tool for solving systems of time dependent partial differential equations in FEniCS.
It achieves this by solving the semidiscretized system of ordinary differential equations arising by applying a finite
element method with a Runge-Kutta method. Gryphon currently supports singly diagonally implicit Runge-Kutta methods with
an explicit first stage (ESDIRKs) of order 2, 3 and 4, developed by Anne Kværnø at the Norwegian University of Science
and Technology.

|
Gryphon is delivered with some (hopefully) easy-to-follow code examples intended to help you on your way to solving your
own problems. The examples included are:

- The heat equation
- The Gray-Scott equations
- The Cahn-Hilliard equations
- The Aliev-Panfilov model (thanks to Vladimir Zverev)

|
Gryphon came about as a result of my masters thesis. It is available in the download section (link to the left).

|
Each branch contains roughly the following folders:

:code:`documentation/`

This folder contains the documentation for Gryphon. To start, open the Gryphon.html file in any web browser (Firefox/Chrome looks good).

:code:`source/`

This folder contains the source code for the Gryphon module. In order to use Gryphon, this folder must be included in your Python-path.

:code:`examplecode/`

This folder contains the Python code for the examples found in the Gryphon user manual. This is a good place to start.

|

How to install
==============
Gryphon requires that FEniCS is installed. Currently only Linux is supported (Ubuntu/Mint tested). Once you have downloaded the desired branch, make sure that the folder :code:`source` is present in your :code:`$PYTHONPATH` variable. When this is in place, you should be able to run the example scripts found in the :code:`examplecode` folder.


|

Announcements
=============

2014-07-26: Gryphon moved to Bitbucket / Gryphon tested on FEniCS 1.4.0 / Small updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
From here on, Gryphon will be hosted on Bitbucket. Also, I have done some preliminary testing with Gryphon on FEniCS 1.4.0, it looks like it should perform as it used to. If you encounter any quirks, let me know. I have also added some small updates to the parameters. The updates include:

- User can specify absolute path for output data
- User can specify which img format the plot of the time steps should have (eps/png/jpg)
- User is prompted when a script is about to reuse an already existing path for storing output data. This is to prevent output data being overwritten (statistics, plots, etc). A parameter has been added to always allow for reuse if desirable.

|
2014-02-10: Gryphon used for science!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Michael J. Welland has recently published an article in the journal Physical Review E, where he used FEniCS and Gryphon to simulate a "Multicomponent phase-field model for extremely large partition coefficients". His article can be found on his website (http://www.mikewelland.com/publications/journalarticle-3) where he has also published the code used for the simulations (http://www.mikewelland.com/publications/code).

Congratulations, Mike!


|
====================================================
Quick introduction to the Gryphon ESDIRK parameters
====================================================

Once an ESDIRK Python object (here denoted as :code:`ESDIRK_Object`) has been created, you can modify the following parameters:

|

Parameters related to runtime behavior of a Gryphon script
==========================================================

:code:`ESDIRK_Object.parameters["drawplot"]`

===============   ==================================================================================================
Possible values	  boolean, True/False
Default value	  False
Effect		  If this value is set to True, a plot of the current time step will be displayed.
		  This can be useful when doing initial testing on a problem.
===============   ==================================================================================================

|
:code:`ESDIRK_Object.parameters["verbose"]`

===============   ==================================================================================================
Possible values	  boolean, True/False
Default value	  False
Effect		  This parameter will cause extra information to be printed to the terminal as the script
		  is running.
===============   ==================================================================================================

|
:code:`ESDIRK_Object.parameters["method"]`

===============   ==================================================================================================
Possible values	  string, "ESDIRK32a"/"ESDIRK32b"/"ESDIRK43a"/"ESDIRK43b"
Default value	  "ESDIRK43b"
Effect		  This parameter selects which ESDIRK method that should be used for the time integration.
		  The default set method "ESDIRK43b" has the highest order and is generally a good place
		  to start.
===============   ==================================================================================================

|
Parameters related to output data from a Gryphon script
=======================================================

:code:`ESDIRK_Object.parameters["output"]["statistics"]`

===============   ==================================================================================================
Possible values	  boolean, True/False
Default value	  False
Effect            If set to True, runtime statistics will be saved to a sub-folder to where the script
                  was run from. The name of the folder will be on the format
		    :code:`[current_working_directory]/<script_name>_gryphon_data`
		  The user can specify an own path by using the parameter "path" (to be defined).
===============   ==================================================================================================

|
:code:`ESDIRK_Object.parameters["output"]["imgformat"]`

===============   ==================================================================================================
Possible values	  string, "eps"/"jpg"/"png"
Default value	  "png"
Effect		  Sets the image format for the plot of the selected time steps by the time stepping
		  algorithm. Note that output statistics must be set to True in order for this to have
		  any effect.
===============   ==================================================================================================

|
:code:`ESDIRK_Object.parameters["output"]["path"]`

===============   =====================================================================================================
Default value	  ""
Effect		  Sets the path where Gryphon output data will be stored. If the path does not start with
		  a slash (/), the folder will be relative to the current working directory. If the path
		  does start with a slash, the path will be absolute. As an example, consider a FEniCS/Gryphon
		  script stored in the folder "/home/user/my_numerical_test/test.py". If the "path" parameter is
		  set to "my_output", the output will be stored in "/home/user/my_numerical_test/my_output/" if the
		  script "test.py" is run from "/home/user/my_numerical_test/". If "path" is set to "/tmp/foo/", the
		  output will be stored in that folder regardless. Gryphon will inform the user if the specified path
		  is unavailable for writing.
===============   =====================================================================================================

|
:code:`ESDIRK_Object.parameters["output"]["plot"]`

===============   ==================================================================================================
Possible values	  boolean, True/False
Default value	  False
Effect		  If this parameter is set to True, a plot of each of the time steps will be stored to a sub folder
		  named "plot" below the path specified by the "path" parameter. The plots will be stored in VTK
                  format.
===============   ==================================================================================================

|
:code:`ESDIRK_Object.parameters["output"]["reuseoutputfolder"]`

===============   ==================================================================================================
Possible values	  boolean, True/False
Default value	  False
Effect		  If the same script is executed twice, any runtime statistics or saved plots are at risk of being
		  overwritten. Because of this, the user must explicitly acknowledge that they want to reuse the
		  specified path whenever a Gryphon script is executed. If this parameter is set to True, Gryphon
		  will reuse the specified path without asking.
===============   ==================================================================================================

|
Parameters related to the time stepping algorithm used in a Gryphon script
==========================================================================

:code:`ESDIRK_Object.parameters["timestepping"]["dt"]`

===============   ========================================================================
Possible values	  double, positive
Default value	  One thousandth of the specified time integration domain.
Effect		  Initial time step used when doing the time integration. If the parameter
                  "adaptive" is set to False, this time step will be used for the entire
                  time integration (fixed time stepping).
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["dtmax"]`

===============   ========================================================================
Possible values	  double, positive
Default value	  0.1
Effect		  Largest allowable time stepping value that Gryphon can use when doing
                  adaptive time stepping.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["dtmin"]`

===============   ========================================================================
Possible values	  double, positive
Default value	  1e-14
Effect		  Smallest allowable time stepping value that Gryphon can use when doing
                  adaptive time stepping.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["adaptive"]`

===============   ========================================================================
Possible values	  boolean, True/False
Default value	  True
Effect		  Turn on/off adaptive time stepping. Setting this to False implies fixed
                  time stepping.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["convergence_criterion"]`

===============   ========================================================================
Possible values   string, "absolute"/"relative"
Default value	  "absolute"
Effect		  Set absolute or relative as convergence criterion when doing adaptive time
                  integration.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["absolute_tolerance"]`

===============   ========================================================================
Possible values	  double, positive
Default value	  1e-07
Effect		  Set the magnitude of the absolute convergence criterion.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["relative_tolerance"]`

===============   ========================================================================
Possible values   double, positive
Default value	  1e-06
Effect		  Set the magnitude of the relative convergence criterion.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["inconsistent_initialdata"]`

===============   ========================================================================
Possible values   boolean, True/False
Default value     False
Effect            If you are using Gryphon to solve an index 1 DAE problem and
		  the initial data is inconsistent, setting this parameter to True
		  will cause Gryphon to take a very small first time step in order to
		  arrive at a better set of initial data before starting the time
		  integration. Note that this might not work if the initial
		  data is too far off.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["stepsizeselector"]`

===============   ========================================================================
Possible values   String, "gustafsson"/"standard"
Default value     "standard"
Effect            Select algorithm for calculating the adaptive step sizes. The standard
		  algorithm only use the previous time step to calculate the next while
		  the Gustafsson algorithm uses the two previous time steps.
===============   ========================================================================

|
:code:`ESDIRK_Object.parameters["timestepping"]["pessimistic_factor"]`

===============   ===============================================
Possible values	  double, [0,1]
Default value	  0.8
Effect		  The pessimistic factor determines how confident
                  Gryphon is in the derived estimate for the next
                  time step. A value of 1.0 is perfect confidence
                  while 0.0 is no confidence (meaningless as the
                  time stepping process will not proceed). This
                  can be tweaked if you are having trouble with
                  too many time steps being rejected.
===============   ===============================================