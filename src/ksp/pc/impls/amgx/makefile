#requirespackage 'PETSC_HAVE_CUDA'
#####requirespackage 'PETSC_HAVE_AMGX'

ALL: lib
CFLAGS  = ${AMGX_INCLUDE}
CFLAGS    =
FFLAGS    =
SOURCEC   = amgx.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/amgx/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
