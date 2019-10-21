/*
  This file defines an interface to the AMGx GPU solver library
*/

// STL
# include <string>
# include <vector>

// AmgX
# include <amgx_c.h>

#include <petsc/private/pcimpl.h>               /*I "petscpc.h" I*/

typedef struct {
  static PetscInt         count;
  AMGX_solver_handle      AMGX = nullptr;
  AMGX_matrix_handle      AmgXA = nullptr;
  AMGX_vector_handle      AmgXP = nullptr;
  AMGX_vector_handle      AmgXRHS = nullptr;
  AMGX_config_handle      cfg = nullptr;
  static AMGX_resources_handle   rsrc;
} PC_AMGX;

// initialize PC_AMGX::count to 0
PetscInt PC_AMGX::count = 0;
// initialize AmgXSolver::rsrc to nullptr;
AMGX_resources_handle AmgXSolver::rsrc = nullptr;

static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PetscErrorCode ierr;
  PC_AMGX        *amgx = (PC_AMGX*)pc->data;
  PetscFunctionBegin;
  // destroy solver instance
  AMGX_solver_destroy(amgx->solver);
  // destroy matrix instance
  AMGX_matrix_destroy(amgx->AmgXA);
  // destroy RHS and unknown vectors
  AMGX_vector_destroy(amgx->AmgXP);
  AMGX_vector_destroy(amgx->AmgXRHS);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  // decrease the number of instances
  // only the last instance need to destroy resource and finalizing AmgX
  if (amgx->count == 1) {
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
  } else {
    AMGX_config_destroy(cfg);
  }
  amgx->count -= 1;
  PetscFunctionReturn(0);
}
static PetscErrorCode PCSetUp_AMGX(PC pc)
{
  PC_AMGX         *amgx = (PC_ASM*)pc->data;
  PetscErrorCode   ierr;





  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_AMGX(PC pc,Vec x,Vec y)
{
  PC_AMGX        *amgx = (PC_AMGX*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;





  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_AMGX(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_AMGX    *jac = (PC_AMGX*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"AMGX options");CHKERRQ(ierr);
  // ierr = PetscOptionsReal("-pc_amgx_lambda","relaxation factor (0 < lambda)","",jac->lambda,&jac->lambda,NULL);CHKERRQ(ierr);
  // ierr = PetscOptionsBool("-pc_amgx_symmetric","apply row projections symmetrically","",jac->symmetric,&jac->symmetric,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_AMGX(PC pc,PetscViewer viewer)
{
  PC_AMGX    *jac = (PC_AMGX*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    // ierr = PetscViewerASCIIPrintf(viewer,"  lambda = %g\n",(double)jac->lambda);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static std::string amgx_arg_str = "config_version=2\n"
"communicator=MPI\n"
"solver(pcgf)=PCG\n"
"determinism_flag=1\n"
"pcgf:preconditioner(prec)=AMG\n"
"pcgf:use_scalar_norm=1\n"
"pcgf:max_iters=10000\n"
"pcgf:convergence=ABSOLUTE\n"
"pcgf:tolerance=1e-8\n"
"pcgf:norm=L2\n"
"pcgf:print_solve_stats=0\n"
"pcgf:monitor_residual=1\n"
"pcgf:obtain_timings=0\n"
"prec:error_scaling=0\n"
"prec:print_grid_stats=1\n"
"prec:max_iters=1\n"
"prec:cycle=V\n"
"prec:min_coarse_rows=2\n"
"prec:max_levels=100\n"
"prec:smoother(smoother)=BLOCK_JACOBI\n"
"prec:presweeps=1\n"
"prec:postsweeps=1\n"
"prec:coarsest_sweeps=1\n"
"prec:coarse_solver(c_solver)=DENSE_LU_SOLVER\n"
"prec:dense_lu_num_rows=2\n"
"prec:algorithm=AGGREGATION\n"
"prec:selector=SIZE_2\n"
"prec:max_matching_iterations=100000\n"
"prec:max_unassigned_percentage=0.0\n"
"smoother:relaxation_factor=0.8\n";

/*MC
     PCAMGX - AMGX iteration

   Options Database Keys:

   Level: beginner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

PETSC_EXTERN PetscErrorCode PCCreate_AMGX(PC pc)
{
  PetscErrorCode ierr;
  PC_AMGX       *amgx;
  AMGX_Mode      mode = dDDI;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&amgx);CHKERRQ(ierr);

  pc->ops->apply           = PCApply_AMGX;
  pc->ops->setfromoptions  = PCSetFromOptions_AMGX;
  pc->ops->setup           = PCSetUp_AMGX;
  pc->ops->view            = PCView_AMGX;
  pc->ops->destroy         = PCDestroy_AMGX;
  pc->data                 = (void*)amgx;
  // increase the number of AmgXSolver instances
  amgx->count += 1;
  // only the first instance (AmgX solver) is in charge of initializing AmgX
  if (amgx->count == 1) {
    // initialize AmgX
    AMGX_SAFE_CALL(AMGX_initialize());
    // intialize AmgX plugings
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    // only the master process can output something on the screen
    AMGX_SAFE_CALL(AMGX_register_print_callback(
                                                [](const char *msg, int length)->void
                                                {PetscPrintf(PetscObjectComm((PetscObject)pc), "%s", msg);}));
    // let AmgX to handle errors returned
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
  }
  // create an AmgX configure object
  AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx->cfg, amgx_arg_str.c_str()));
  // let AmgX handle returned error codes internally
  AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "exception_handling=1"));
  // create an AmgX resource object, only the first instance is in charge
  if (count == 1) AMGX_resources_create(&amgx->rsrc, amgx->cfg, &gpuWorld, 1, &devID);
  // create AmgX vector object for unknowns and RHS
  AMGX_vector_create(&amgx->AmgXP, amgx->rsrc, mode);
  AMGX_vector_create(&amgx->AmgXRHS, amgx->rsrc, mode);
  // create AmgX matrix object for unknowns and RHS
  AMGX_matrix_create(&amgx->AmgXA, amgx->rsrc, mode);
  // create an AmgX solver object
  AMGX_solver_create(&amgx->AMGX, amgx->rsrc, mode, amgx->cfg);
  // obtain the default number of rings based on current configuration
  AMGX_config_get_default_number_of_rings(amgx->cfg, &ring);

  PetscFunctionReturn(0);
}
