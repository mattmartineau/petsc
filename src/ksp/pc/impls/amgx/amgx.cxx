/*
  This file implements a PC interface to the AMGx GPU solver library
*/
#include <petsc/private/pcimpl.h>               /*I "petscpc.h" I*/

// STL
# include <string>
# include <vector>

// CUDA
# include <cuda_runtime.h>

// AmgX
# include <amgx_c.h>

typedef struct _PC_AMGX {
  static PetscInt         count;
  AMGX_solver_handle      AMGX;
  AMGX_matrix_handle      AmgXA;
  AMGX_vector_handle      AmgXP;
  AMGX_vector_handle      AmgXRHS;
  AMGX_config_handle      cfg;
  static AMGX_resources_handle   rsrc;
} PC_AMGX;

// initialize PC_AMGX::count to 0
PetscInt PC_AMGX::count = 0;
// initialize AmgXSolver::rsrc to nullptr;
AMGX_resources_handle PC_AMGX::rsrc = nullptr;

static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PetscErrorCode ierr;
  PC_AMGX        *amgx = (PC_AMGX*)pc->data;
  PetscFunctionBegin;
  // destroy solver instance
  AMGX_solver_destroy(amgx->AMGX);
  // destroy matrix instance
  AMGX_matrix_destroy(amgx->AmgXA);
  // destroy RHS and unknown vectors
  AMGX_vector_destroy(amgx->AmgXP);
  AMGX_vector_destroy(amgx->AmgXRHS);
  // decrease the number of instances
  // only the last instance need to destroy resource and finalizing AmgX
  if (amgx->count == 1) {
    AMGX_resources_destroy(amgx->rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(amgx->cfg));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
  } else {
    AMGX_config_destroy(amgx->cfg);
  }
  amgx->count -= 1;
  ierr = PetscFree(amgx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PCSetUp_AMGX(PC pc)
{
  PC_AMGX         *amgx = (PC_AMGX*)pc->data;
  PetscErrorCode   ierr;
  Mat              Pmat = pc->pmat, localA;
  PetscInt         Iend,nGlobalRows,nLocalRows,rawN;
  AMGX_distribution_handle dist;
  std::vector<PetscInt>    partData;
  std::vector<PetscInt64>  offsets;
  int                      nranks;
  const PetscInt          *rawCol, *rawRow;
  PetscScalar             *rawData;
  PetscBool                done;
  std::vector<int>         row;
  std::vector<PetscInt64>  col;
  std::vector<PetscScalar> data;
  PetscFunctionBegin;
  if (!pc->setupcalled) {
    int              ring;
    // create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&amgx->AmgXP, amgx->rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx->AmgXRHS, amgx->rsrc, AMGX_mode_dDDI);
    // create AmgX matrix object for unknowns and RHS
    AMGX_matrix_create(&amgx->AmgXA, amgx->rsrc, AMGX_mode_dDDI);
    // create an AmgX solver object
    AMGX_solver_create(&amgx->AMGX, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg);
    // obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(amgx->cfg, &ring); // should be 2 for classical, 1 for else
  }
  // upload matrix
  AMGX_distribution_create(&dist, amgx->cfg);
  // get offsets
  ierr = MatGetOwnershipRange(Pmat,NULL,&Iend);CHKERRQ(ierr);
  Iend++; // add 1 to allow reusing gathered ismax values as partition offsets for AMGX
  MPI_Comm_size(PetscObjectComm((PetscObject)pc), &nranks);
  partData.resize(nranks);
  ierr = MPI_Allgather(&Iend, 1, MPIU_INT, &partData[0], 1, MPIU_INT, PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  partData.insert(partData.begin(), 0); // partition 0 always starts at 0
  offsets.assign(partData.begin(), partData.end());
  AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, offsets.data());
  // upload matrix
  ierr = MatGetSize(Pmat, &nGlobalRows, nullptr); CHKERRQ(ierr);
  ierr = MatGetLocalSize(Pmat, &nLocalRows, nullptr); CHKERRQ(ierr);
  // get local matrix from redistributed matrix
  ierr = MatMPIAIJGetLocalMat(Pmat, MAT_INITIAL_MATRIX, &localA);CHKERRQ(ierr);
  ierr = MatGetRowIJ(localA, 0, PETSC_FALSE, PETSC_FALSE, &rawN, &rawRow, &rawCol, &done);CHKERRQ(ierr);
  if (rawN!=nLocalRows)SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_SIG, "rawN!=nLocalRows %D %D\n",rawN,nLocalRows);
  // check if the function worked
  if (!done) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SIG, "MatGetRowIJ did not work!\n");
  // get data
  ierr = MatSeqAIJGetArray(localA, &rawData);CHKERRQ(ierr);
  // copy values to STL vector. Note: there is an implicit conversion from
  // PetscInt to PetscInt64 for the column vector
  col.assign(rawCol, rawCol+rawRow[nLocalRows]);
  row.assign(rawRow, rawRow+nLocalRows+1);
  data.assign(rawData, rawData+rawRow[nLocalRows]);
  // return ownership of memory space to PETSc
  ierr = MatRestoreRowIJ(localA, 0, PETSC_FALSE, PETSC_FALSE, &rawN, &rawRow, &rawCol, &done);CHKERRQ(ierr);
  // check if the function worked
  if (!done) SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_SIG, "MatRestoreRowIJ did not work!");
  // return ownership of memory space to PETSc
  ierr = MatSeqAIJRestoreArray(localA, &rawData);CHKERRQ(ierr);
  if (localA != Pmat) {
    ierr = MatDestroy(&localA);CHKERRQ(ierr);
  }
  // upload
  AMGX_matrix_upload_distributed(amgx->AmgXA, nGlobalRows, nLocalRows, row[nLocalRows],
                                 1, 1, row.data(), col.data(), data.data(),
                                 nullptr, dist);
  AMGX_distribution_destroy(dist);

  // bind the matrix A to the solver
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  AMGX_solver_setup(amgx->AMGX, amgx->AmgXA);

  // connect (bind) vectors to the matrix
  AMGX_vector_bind(amgx->AmgXP, amgx->AmgXA);
  AMGX_vector_bind(amgx->AmgXRHS, amgx->AmgXA);

  PetscFunctionReturn(0);
}
static PetscErrorCode PCApply_AMGX(PC pc,Vec p,Vec b)
{
  PC_AMGX          *amgx = (PC_AMGX*)pc->data;
  PetscErrorCode    ierr;
  double           *unks, *rhs;
  PetscInt          size;
  AMGX_SOLVE_STATUS status;
  PetscFunctionBegin;
  // get size of local vector (p and b should have the same local size)
  ierr = VecGetLocalSize(p, &size); CHKERRQ(ierr);
  // get pointers to the raw data of local vectors
  ierr = VecGetArray(p, &unks); CHKERRQ(ierr);
  ierr = VecGetArray(b, &rhs); CHKERRQ(ierr);
  // upload vectors to AmgX
  AMGX_vector_upload(amgx->AmgXP, size, 1, unks);
  AMGX_vector_upload(amgx->AmgXRHS, size, 1, rhs);
  // solve
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)pc)); CHKERRQ(ierr);
  AMGX_solver_solve(amgx->AMGX, amgx->AmgXRHS, amgx->AmgXP);
  // get the status of the solver
  AMGX_solver_get_status(amgx->AMGX, &status);
  // check whether the solver successfully solve the problem
  if (status != AMGX_SOLVE_SUCCESS) SETERRQ1(PetscObjectComm((PetscObject)pc),
                                             PETSC_ERR_CONV_FAILED, "AmgX solver failed to solve the system! "
                                             "The error code is %d.\n", status);
  // download data from device
  AMGX_vector_download(amgx->AmgXP, unks);
  // restore PETSc vectors
  ierr = VecRestoreArray(p, &unks); CHKERRQ(ierr);
  ierr = VecRestoreArray(b, &rhs); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_AMGX(PetscOptionItems *PetscOptionsObject,PC pc)
{
  //PC_AMGX    *amgx = (PC_AMGX*)pc->data;
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
  //PC_AMGX       *amgx = (PC_AMGX*)pc->data;
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
  cudaError_t    err = cudaSuccess;
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
    // intialize AmgX plugins
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    // only the master process can output something on the screen
    AMGX_SAFE_CALL(AMGX_register_print_callback([]
                                                (const char *msg, int length)->void{
                                                  PetscPrintf(PETSC_COMM_WORLD, "%s", msg);}));
    // let AmgX to handle errors returned
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
  }
  // create an AmgX configure object
  AMGX_config_create(&amgx->cfg, "communicator=MPI");
  // let AmgX handle returned error codes internally
  AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "exception_handling=1"));
  AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "solver(mg)=AMG"));
  AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "mg:algorithm=AGGREGATION"));
  // create an AmgX resource object, only the first instance is in charge
  if (amgx->count == 1) {
    int devID,devCount;
    MPI_Comm comm = PetscObjectComm((PetscObject)pc);
    err = cudaGetDevice(&devID);
    if (err != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in cudaGetDevice %s",cudaGetErrorString(err));
    AMGX_resources_create(&amgx->rsrc, amgx->cfg, &comm, 1, &devID);
    err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in cudaGetDeviceCount %s",cudaGetErrorString(err));
    if (devCount!=1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error devCount %d != 1",devCount);
  }

  PetscFunctionReturn(0);
}
