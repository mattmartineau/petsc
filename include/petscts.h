/* $Id: ts.h,v 1.18 1997/10/12 21:59:37 bsmith Exp bsmith $ */
/*
   User interface for the timestepping package. This is package
   is for use in solving time-dependent PDEs.
*/
#if !defined(__TS_PACKAGE)
#define __TS_PACKAGE
#include "snes.h"

typedef struct _p_TS* TS;
#define TS_COOKIE PETSC_COOKIE+18

typedef enum { TS_EULER, TS_BEULER, TS_PSEUDO, TS_PVODE, TS_NEW} TSType;
typedef enum { TS_LINEAR, TS_NONLINEAR} TSProblemType;

extern int TSCreate(MPI_Comm,TSProblemType,TS*);
extern int TSSetType(TS,TSType);
extern int TSGetProblemType(TS,TSProblemType*);
extern int TSDestroy(TS);

extern int TSSetMonitor(TS,int(*)(TS,int,double,Vec,void*),void *);
extern int TSGetType(TS,TSType*,char**);

extern int TSSetOptionsPrefix(TS, char *);
extern int TSAppendOptionsPrefix(TS, char *);
extern int TSGetOptionsPrefix(TS, char **);
extern int TSSetFromOptions(TS);
extern int TSSetUp(TS);

extern int TSSetSolution(TS,Vec);
extern int TSGetSolution(TS,Vec*);

extern int TSSetDuration(TS,int,double);
extern int TSPrintHelp(TS);

extern int TSDefaultMonitor(TS,int,double,Vec,void*);
extern int TSStep(TS,int *,double*);

extern int TSSetInitialTimeStep(TS,double,double);
extern int TSGetTimeStep(TS,double*);
extern int TSGetTime(TS,double*);
extern int TSGetTimeStepNumber(TS,int*);
extern int TSSetTimeStep(TS,double);

extern int TSSetRHSFunction(TS,int (*)(TS,double,Vec,Vec,void*),void*);
extern int TSSetRHSMatrix(TS,Mat,Mat,int (*)(TS,double,Mat*,Mat*,MatStructure*,void*),void*);
extern int TSSetRHSJacobian(TS,Mat,Mat,int(*)(TS,double,Vec,Mat*,Mat*,MatStructure*,void*),void*);

extern int TSDefaultComputeJacobianWithColoring(TS,double,Vec,Mat*,Mat*,MatStructure*,void*);
extern int TSSetRHSJacobianDefault(TS,MatFDColoring,Mat,Mat);
extern int TSDefaultComputeJacobianSlow(TS,double,Vec,Mat*,Mat*,MatStructure*,void*);

extern int TSGetRHSMatrix(TS,Mat*,Mat*,void**);
extern int TSGetRHSJacobian(TS,Mat*,Mat*,void**);

extern int TSPseudoSetTimeStep(TS,int(*)(TS,double*,void*),void*);
extern int TSPseudoDefaultTimeStep(TS,double*,void* );
extern int TSPseudoComputeTimeStep(TS,double *);

extern int TSPseudoSetVerifyTimeStep(TS,int(*)(TS,Vec,void*,double*,int*),void*);
extern int TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,double*,int*);
extern int TSPseudoVerifyTimeStep(TS,Vec,double*,int*);
extern int TSPseudoSetTimeStepIncrement(TS,double);
extern int TSPseudoIncrementDtFromInitialDt(TS);

extern int TSComputeRHSFunction(TS,double,Vec,Vec);

extern int TSRegister(TSType,TSType*,char*,int (*)(TS));
extern int TSRegisterAll();
extern int TSRegisterDestroy();
extern int TSRegisterAllCalled;

extern int TSGetSNES(TS,SNES*);
extern int TSGetSLES(TS,SLES*);

extern int TSView(TS,Viewer);

extern int TSSetApplicationContext(TS,void *);
extern int TSGetApplicationContext(TS,void **);

extern int TSLGMonitorCreate(char *,char *,int,int,int,int, DrawLG *);
extern int TSLGMonitor(TS,int,double,Vec,void *);
extern int TSLGMonitorDestroy(DrawLG);

/*
       PETSc interface to PVode
*/
typedef enum { PVODE_ADAMS, PVODE_BDF } TSPVodeType;
extern int TSPVodeSetType(TS,TSPVodeType);
extern int TSPVodeGetPC(TS,PC*);

#endif





