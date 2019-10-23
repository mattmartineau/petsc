#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS    = ${AMGX_INCLUDE}
FFLAGS    =
SOURCECXX = amgx.cpp
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/amgx/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
