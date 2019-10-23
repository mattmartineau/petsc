####requirespackage 'PETSC_HAVE_CUDA'
####requirespackage 'PETSC_HAVE_AMGX'

ALL: lib
CXXFLAGS  = ${AMGX_INCLUDE}
CFLAGS    =
FFLAGS    =
SOURCECXX = amgx.cxx
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/amgx/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
