#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: redundant.c,v 1.3 1999/03/26 18:41:21 bsmith Exp balay $";
#endif
/*
  This file defines a "solve the problem redundantly on each processor" preconditioner.

*/
#include "src/sles/pc/pcimpl.h"     /*I "pc.h" I*/
#include "sles.h"

typedef struct {
  PC         pc;                    /* actual preconditioner used on each processor */
  Vec        x,b;                   /* sequential vectors to hold parallel vectors */
  Mat        *mats,*pmats;          /* matrix and optional preconditioner matrix */
  VecScatter scatterin,scatterout;  /* scatter used to move all values to each processor */
  PetscTruth useparallelmat;
} PC_Redundant;

#undef __FUNC__  
#define __FUNC__ "PCView_Redundant"
static int PCView_Redundant(PC pc,Viewer viewer)
{
  PC_Redundant  *red = (PC_Redundant *) pc->data;
  int           ierr;
  ViewerType    vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ViewerASCIIPrintf(viewer,"  Redundant solver preconditioner: Actual PC follows\n");
    ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PCView(red->pc,viewer); CHKERRQ(ierr);
    ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,STRING_VIEWER)) {
    ViewerStringSPrintf(viewer," Redundant solver preconditioner");
    ierr = PCView(red->pc,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_Redundant"
static int PCSetUp_Redundant(PC pc)
{
  PC_Redundant   *red  = (PC_Redundant *) pc->data;
  int            ierr,mstart,mlocal,m;
  IS             isl;
  MatReuse       reuse = MAT_INITIAL_MATRIX;
  MatStructure   str   = DIFFERENT_NONZERO_PATTERN;

  PetscFunctionBegin;
  ierr = VecGetSize(pc->vec,&m);CHKERRQ(ierr);
  if (pc->setupcalled == 0) {
    /*
       Create the vectors and vector scatter to get the entire vector onto each processor
    */
    ierr = VecGetLocalSize(pc->vec,&mlocal);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(pc->vec,&mstart,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,m,&red->x);CHKERRQ(ierr);
    ierr = VecDuplicate(red->x,&red->b);CHKERRQ(ierr);
    ierr = PCSetVector(red->pc,red->x);CHKERRQ(ierr);

    ierr = VecScatterCreate(pc->vec,0,red->x,0,&red->scatterin);CHKERRQ(ierr);

    ierr = ISCreateStride(pc->comm,mlocal,mstart,1,&isl);CHKERRQ(ierr);
    ierr = VecScatterCreate(red->x,isl,pc->vec,isl,&red->scatterout);CHKERRQ(ierr);
    ierr = ISDestroy(isl);CHKERRQ(ierr);
  }

  if (red->useparallelmat) {
    if (pc->setupcalled == 1 && pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrices */
      if (red->pmats && red->pmats != red->mats) {
        ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
      }
      if (red->mats) {
        ierr = MatDestroyMatrices(1,&red->mats);CHKERRQ(ierr);
      }   
    } else if (pc->setupcalled == 1) {
      reuse = MAT_REUSE_MATRIX;
      str   = SAME_NONZERO_PATTERN;
    }
        
    /* 
       grab the parallel matrix and put it on each processor
    */
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isl);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(pc->mat,1,&isl,&isl,reuse,&red->mats);CHKERRQ(ierr);
    if (pc->pmat != pc->mat) {
      ierr = MatGetSubMatrices(pc->pmat,1,&isl,&isl,reuse,&red->pmats);CHKERRQ(ierr);
    } else {
      red->pmats = red->mats;
    }
    ierr = ISDestroy(isl);CHKERRQ(ierr);

    /* tell sequential PC its operators */
    ierr = PCSetOperators(red->pc,red->mats[0],red->pmats[0],str);CHKERRQ(ierr);
  }    

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCApply_Redundant"
static int PCApply_Redundant(PC pc,Vec x,Vec y)
{
  PC_Redundant      *red = (PC_Redundant *) pc->data;
  int               ierr;

  PetscFunctionBegin;
  /* move all values to each processor */
  ierr = VecScatterBegin(x,red->b,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,red->b,INSERT_VALUES,SCATTER_FORWARD,red->scatterin);CHKERRQ(ierr);

  /* apply preconditioner on each processor */
  ierr = PCApply(red->pc,red->b,red->x);CHKERRQ(ierr);

  /* move local part of values into y vector */
  ierr = VecScatterBegin(red->x,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  ierr = VecScatterEnd(red->x,y,INSERT_VALUES,SCATTER_FORWARD,red->scatterout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCDestroy_Redundant"
static int PCDestroy_Redundant(PC pc)
{
  PC_Redundant *red = (PC_Redundant *) pc->data;
  int          ierr;

  PetscFunctionBegin;
  if (red->scatterin)  {ierr = VecScatterDestroy(red->scatterin);CHKERRQ(ierr);}
  if (red->scatterout) {ierr = VecScatterDestroy(red->scatterout);CHKERRQ(ierr);}
  if (red->x)          {ierr = VecDestroy(red->x);CHKERRQ(ierr);}
  if (red->b)          {ierr = VecDestroy(red->b);CHKERRQ(ierr);}
  if (red->pmats && red->pmats != red->mats) {
    ierr = MatDestroyMatrices(1,&red->pmats);CHKERRQ(ierr);
  }
  if (red->mats) {
    ierr = MatDestroyMatrices(1,&red->mats);CHKERRQ(ierr);
  }
  ierr = PCDestroy(red->pc);CHKERRQ(ierr);
  PetscFree(red);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_Redundant"
static int PCPrintHelp_Redundant(PC pc,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(pc->comm," Options for PCRedundant preconditioner:\n");
  (*PetscHelpPrintf)(pc->comm," %sredundant : prefix to control options for redundant PC.\
  Add before the \n      usual PC option names (e.g., %sredundant_pc_type\
  <method>)\n",p,p);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_Redundant"
static int PCSetFromOptions_Redundant(PC pc)
{
  int          ierr;
  PC_Redundant *red = (PC_Redundant *) pc->data;

  PetscFunctionBegin;
  ierr = PCSetFromOptions(red->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_Redundant"
int PCCreate_Redundant(PC pc)
{
  int          ierr;
  PC_Redundant *red;
  char         *prefix;

  PetscFunctionBegin;
  red = PetscNew(PC_Redundant); CHKPTRQ(red);
  PLogObjectMemory(pc,sizeof(PC_Redundant));
  PetscMemzero(red,sizeof(PC_Redundant)); 
  red->useparallelmat   = PETSC_TRUE;

  /* create the sequential PC that each processor has copy of */
  ierr = PCCreate(PETSC_COMM_SELF,&red->pc);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix); CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(red->pc,prefix); CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(red->pc,"redundant_"); CHKERRQ(ierr);

  pc->apply             = PCApply_Redundant;
  pc->applytrans        = 0;
  pc->setup             = PCSetUp_Redundant;
  pc->destroy           = PCDestroy_Redundant;
  pc->printhelp         = PCPrintHelp_Redundant;
  pc->setfromoptions    = PCSetFromOptions_Redundant;
  pc->setuponblocks     = 0;
  pc->data              = (void *) red;
  pc->view              = PCView_Redundant;
  pc->applyrich         = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
