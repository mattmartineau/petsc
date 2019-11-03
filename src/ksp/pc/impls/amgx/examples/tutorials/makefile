
# This directory contains PC/AMGX example programs for solving system

CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/ksp/pc/impls/amgx/examples/tutorials/
MANSEC           = PC
EXAMPLESC	 = ex1.c
EXAMPLESF	 =
EXAMPLESCH	 =
EXAMPLESFH       =
EXAMPLESMATLAB   =
DIRS		 =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#-------------------------------------------------------------------------

#  these tests are used by the makefile in PETSC_DIR for basic tests of the install and should not be removed
run:
	-@${MPIEXEC} -n 2 ./ex1 -m /ccs/home/adams/AMGX/examples/matrix.mtx -c /ccs/home/adams/AMGX/core/configs/FGMRES_AGGREGATION.json

runex1:
	-@${MPIEXEC} -n 2 ./ex1 -m /ccs/home/adams/AMGX/examples/matrix.mtx -c /ccs/home/adams/AMGX/core/configs/FGMRES_AGGREGATION.json > ex1_1.tmp 2>&1; \
	   if (${DIFF} output/ex1_ml.out ex1_1.tmp) then  \
           echo "C/C++ example src/snes/examples/tutorials/ex1 run successfully with ml"; \
           else  printf "${PWD}\nPossible problem with ex1 running with ml, diffs above\n=========================================\n"; fi; \
           ${RM} -f ex1_1.tmp

include ${PETSC_DIR}/lib/petsc/conf/test
