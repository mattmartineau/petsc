/*
  This file implements a PC interface to the AMGx CUDA GPU solver library
*/
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

#include "cuda_runtime.h"

// AmgX
#include <amgx_c.h>

#define AMGXDEBUG

typedef struct _PC_AMGX
{
    AMGX_solver_handle solver;
    AMGX_matrix_handle A;
    AMGX_vector_handle P;
    AMGX_vector_handle RHS;
    AMGX_config_handle cfg;
    MPI_Comm comm;
    AMGX_resources_handle rsrc;
    void *lib_handle;
    char filename[PETSC_MAX_PATH_LEN];

    // Cached state for re-setup
    int nnz;
    int nLocalRows;
    Mat localA;
    PetscScalar *values;

} PC_AMGX;
static PetscInt s_count = 0;

/* ----------------------------------------------------------------------------- */
PetscErrorCode PCReset_AMGX(PC pc)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif

    PC_AMGX *amgx = (PC_AMGX *)pc->data;

    PetscFunctionBegin;
    AMGX_solver_destroy(amgx->solver);
    AMGX_matrix_destroy(amgx->A);
    AMGX_vector_destroy(amgx->P);
    AMGX_vector_destroy(amgx->RHS);
    PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_AMGX(PC pc)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif

    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    cudaError_t err = cudaSuccess;

    PetscFunctionBegin;

    // XXX I am not sure it is a good idea to automatically call reset here
    // as it seems to be called internally by PETSc on destroy?
    //ierr = PCReset(pc);
    //CHKERRQ(ierr);

    /* decrease the number of instances, only the last instance need to destroy resource and finalizing AmgX */
    if (s_count == 1)
    { /* can put this in a PCAMGXInitializePackage method */
        if (!amgx->rsrc)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "s_rsrc == NULL");
        err = AMGX_resources_destroy(amgx->rsrc);
        if (err != cudaSuccess)
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "error: %s", cudaGetErrorString(err));
        /* destroy config (need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_destroy(amgx->cfg));
        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
        PetscErrorCode ierr = MPI_Comm_free(&amgx->comm);
        CHKERRQ(ierr);
#ifdef AMGX_DYNAMIC_LOADING
        amgx_libclose(amgx->lib_handle);
#endif
    }
    else
    {
        AMGX_SAFE_CALL(AMGX_config_destroy(amgx->cfg));
    }
    s_count -= 1;
    PetscErrorCode ierr = PetscFree(amgx);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

void printMemory()
{
    size_t freeB;
    size_t totalB;
    cudaMemGetInfo(&freeB, &totalB);
    double freeMB = (double)freeB / 1024.0 / 1024.0;
    double totalMB = (double)totalB / 1024.0 / 1024.0;
    double usedMB = totalMB - freeMB;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", usedMB, freeMB, totalMB);
}

static PetscErrorCode PCSetUp_AMGX(PC pc)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif
    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    Mat Pmat = pc->pmat;

    PetscFunctionBegin;

    if (!pc->setupcalled)
    {
#ifdef AMGXDEBUG
        printMemory();
#endif

        AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx->cfg, amgx->filename));

        /* switch on internal error handling (no need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "exception_handling=1"));

        // XXX Unless I missed something outside of this file, must change for multi GPU I would expect
        int devID, devCount;

        cudaError_t err = cudaGetDevice(&devID);
        if (err != cudaSuccess)
        {
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "error in cudaGetDevice %s", cudaGetErrorString(err));
        }

        err = cudaGetDeviceCount(&devCount);
        if (err != cudaSuccess)
        {
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "error in cudaGetDeviceCount %s", cudaGetErrorString(err));
        }

        /* create an AmgX resource object, only the first instance is in charge */

        err = AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &devID);
        err = AMGX_matrix_create(&amgx->A, amgx->rsrc, AMGX_mode_dDDI);
        err = AMGX_vector_create(&amgx->P, amgx->rsrc, AMGX_mode_dDDI);
        err = AMGX_vector_create(&amgx->RHS, amgx->rsrc, AMGX_mode_dDDI);
        err = AMGX_solver_create(&amgx->solver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg);

        /* upload matrix */
        int nranks;
        int rank;
        MPI_Comm_size(amgx->comm, &nranks);
        MPI_Comm_rank(amgx->comm, &rank);

        PetscInt nGlobalRows;
        PetscErrorCode ierr = MatGetSize(Pmat, &nGlobalRows, NULL);
        CHKERRQ(ierr);

        // XXX This isn't a requirement on AmgX, rather this integration doesn't yet handle 64bit indices correctly
        if (nGlobalRows >= 2147483648)
        {
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMGx only supports 32 bit ints. N = %D", nGlobalRows);
        }

        ierr = MatGetLocalSize(Pmat, &amgx->nLocalRows, NULL);
        CHKERRQ(ierr);

        PetscInt bs;
        ierr = MatGetBlockSize(Pmat, &bs);
        CHKERRQ(ierr);

        AMGX_distribution_handle dist;

        /* to calculate the partition offsets and pass those into the API call instead of creating a full partition vector. */
        PetscInt *partitionOffsets;

        /* get raw matrix data */
        const PetscInt *colIndices, *rowOffsets;
        PetscScalar *values;

        // Need some robust check to determine if the matrix is an AmgX matrix
        PetscBool isAmgXMatrix;
        ierr = PetscObjectTypeCompare((PetscObject)Pmat, MATSEQAIJ, &isAmgXMatrix);
        CHKERRQ(ierr);

        // At the present time, an AmgX matrix is a sequential matrix
        // Non-sequential/MPI matrices must be adapted to extract the local matrix
        if (isAmgXMatrix || nranks == 1)
        {
            amgx->localA = Pmat;
        }
        else
        {
            ierr = MatMPIAIJGetLocalMat(Pmat, MAT_INITIAL_MATRIX, &amgx->localA);
            CHKERRQ(ierr);
        }

        // Extract the CSR data
        PetscInt rawN;
        PetscBool done;
        ierr = MatGetRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &rawN, &rowOffsets, &colIndices, &done);
        CHKERRQ(ierr);

        if (!done)
        {
            SETERRQ(amgx->comm, PETSC_ERR_PLIB, "MatGetRowIJ was not successful\n");
        }
        if (rawN != amgx->nLocalRows)
        {
            SETERRQ2(amgx->comm, PETSC_ERR_PLIB, "rawN != nLocalRows %D %D\n", rawN, amgx->nLocalRows);
        }

        ierr = MatSeqAIJGetArray(amgx->localA, &values);
        CHKERRQ(ierr);

        if (isAmgXMatrix)
        {
            cudaMemcpy(&amgx->nnz, &rowOffsets[amgx->nLocalRows], sizeof(int), cudaMemcpyDefault);
        }
        else
        {
            amgx->nnz = rowOffsets[amgx->nLocalRows];
        }

        ierr = PetscMalloc1(nranks + 1, &partitionOffsets);
        CHKERRQ(ierr);

        partitionOffsets[0] = 0; /* could use PetscLayoutGetRanges */

        ierr = MPI_Allgather(&amgx->nLocalRows, sizeof(PetscInt), MPI_BYTE, &partitionOffsets[1], sizeof(PetscInt), MPI_BYTE, amgx->comm);
        CHKERRQ(ierr);

        for (int i = 1; i <= nranks; i++)
        {
            partitionOffsets[i] += partitionOffsets[i - 1];
        }

        int nGlobal = partitionOffsets[nranks]; // last element always has global number of rows

        /* upload - this takes an int for nglobal (does it work for large sysetms ??) */
        int petsc32 = (sizeof(PetscInt) == 4);
        AMGX_distribution_create(&dist, amgx->cfg);
        AMGX_distribution_set_32bit_colindices(dist, petsc32);
        AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partitionOffsets);
        AMGX_matrix_upload_distributed(
            amgx->A, nGlobal, amgx->nLocalRows, amgx->nnz, bs, bs,
            rowOffsets, colIndices, values, NULL, dist);
        AMGX_distribution_destroy(dist);

        ierr = PetscFree(partitionOffsets);
        CHKERRQ(ierr);

        /* bind the matrix A to the solver */
        ierr = MPI_Barrier(amgx->comm);
        CHKERRQ(ierr);

        AMGX_solver_setup(amgx->solver, amgx->A);

        /* connect (bind) vectors to the matrix */
        AMGX_vector_bind(amgx->P, amgx->A);
        AMGX_vector_bind(amgx->RHS, amgx->A);
    }
    else
    {
#ifdef AMGXDEBUG
        printMemory();
#endif

        // The fast path after the initial setup phase
        AMGX_matrix_replace_coefficients(amgx->A, amgx->nLocalRows, amgx->nnz, amgx->values, NULL);

        AMGX_solver_resetup(amgx->solver, amgx->A);
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_AMGX(PC pc, Vec b, Vec x)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif

    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    PetscScalar *unks;
    const PetscScalar *rhs;
    PetscInt n;
    AMGX_SOLVE_STATUS status;

    PetscFunctionBegin;

    PetscErrorCode ierr = VecGetLocalSize(x, &n);
    CHKERRQ(ierr);
    ierr = VecGetArray(x, &unks);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(b, &rhs);
    CHKERRQ(ierr);

    AMGX_vector_upload(amgx->P, n, 1, unks);
    AMGX_vector_upload(amgx->RHS, n, 1, rhs);

    ierr = MPI_Barrier(amgx->comm);
    CHKERRQ(ierr);

    AMGX_solver_solve(amgx->solver, amgx->RHS, amgx->P);
    AMGX_solver_get_status(amgx->solver, &status);

    if (status == AMGX_SOLVE_FAILED)
    {
        SETERRQ1(amgx->comm, PETSC_ERR_CONV_FAILED,
                 "AmgX solver failed to solve the system! "
                 "The error code is %d.\n",
                 status);
    }

    AMGX_vector_download(amgx->P, unks);

    ierr = VecRestoreArray(x, &unks);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(b, &rhs);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_AMGX(PetscOptionItems *PetscOptionsObject, PC pc)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif

    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    PetscErrorCode ierr;
    PetscBool exists;

    PetscFunctionBegin;
    ierr = PetscOptionsHead(PetscOptionsObject, "AMGX options");
    CHKERRQ(ierr);
    ierr = PetscOptionsString("-pc_amgx_json", "AMGX parameter file (json)", "amgx.c", amgx->filename, amgx->filename, PETSC_MAX_PATH_LEN, NULL);
    CHKERRQ(ierr);
    ierr = PetscStrreplace(PetscObjectComm((PetscObject)pc), amgx->filename, amgx->filename, PETSC_MAX_PATH_LEN);
    CHKERRQ(ierr);
    ierr = PetscTestFile(amgx->filename, 'r', &exists);
    CHKERRQ(ierr);

    if (!exists)
    {
        printf("Parameter -pc_amgx_json incorrect.\n");

        /* try to add prefix */
        char str[PETSC_MAX_PATH_LEN];
        ierr = PetscSNPrintf(str, PETSC_MAX_PATH_LEN - 1, "${PETSC_DIR}/share/petsc/amgx/%s", amgx->filename);
        CHKERRQ(ierr);
        ierr = PetscStrreplace(PetscObjectComm((PetscObject)pc), str, amgx->filename, PETSC_MAX_PATH_LEN);
        CHKERRQ(ierr);
        ierr = PetscTestFile(amgx->filename, 'r', &exists);
        CHKERRQ(ierr);

        if (!exists)
        {
            SETERRQ1(PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "input file not found (%s)", amgx->filename);
        }
    }
    else
    {
        printf("As per -pc_amgx_json, found parameter file at %s.\n", amgx->filename);
    }

    ierr = PetscOptionsTail();
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode PCView_AMGX(PC pc, PetscViewer viewer)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif
    PetscErrorCode ierr;
    PetscBool iascii;

    PetscFunctionBegin;
    ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii);
    CHKERRQ(ierr);
    if (iascii)
    {
    }
    PetscFunctionReturn(0);
}

/* print callback (could be customized) */
static void print_callback(const char *msg, int length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        printf("%s", msg);
    }
}

/*MC
 PCAMGX - AMGX iteration

 Options Database Keys:

 Level: beginner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/
PETSC_EXTERN PetscErrorCode PCCreate_AMGX(PC pc)
{
#ifdef AMGXDEBUG
    printf("in %s\n", __func__);
#endif

    PC_AMGX *amgx;

    PetscFunctionBegin;
    PetscErrorCode ierr = PetscNewLog(pc, &amgx);
    CHKERRQ(ierr);
    pc->ops->apply = PCApply_AMGX;
    pc->ops->setfromoptions = PCSetFromOptions_AMGX;
    pc->ops->setup = PCSetUp_AMGX;
    pc->ops->view = PCView_AMGX;
    pc->ops->destroy = PCDestroy_AMGX;
    pc->ops->reset = PCReset_AMGX;
    pc->data = (void *)amgx;
    s_count += 1;
    if (s_count == 1)
    {   /* can put this in a PCAMGXFinalizePackage method */
        /* load the library (if it was dynamically loaded) */
#ifdef AMGX_DYNAMIC_LOADING
        amgx->lib_handle = NULL;
#ifdef _WIN32
        amgx->lib_handle = amgx_libopen("amgxsh.dll");
#else
        printf("dynamic loading\n");
        amgx->lib_handle = amgx_libopen("libamgxsh.so");
#endif
        if (amgx->lib_handle == NULL)
        {
            errAndExit("ERROR: can not load the library");
        }
        //load all the routines
        if (amgx_liblink_all(amgx->lib_handle) == 0)
        {
            amgx_libclose(amgx->lib_handle);
            errAndExit("ERROR: corrupted library loaded\n");
        }
#endif
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }
    {
        MPI_Comm comm_in = PetscObjectComm((PetscObject)pc);
        /* This communicator is not yet known to this system, so we duplicate it and make an internal communicator */
        ierr = MPI_Comm_dup(comm_in, &amgx->comm);
        CHKERRQ(ierr);
    }
    /* set a default path/filename, use -pc_amgx_json to set at runtime */
    ierr = PetscSNPrintf(amgx->filename, PETSC_MAX_PATH_LEN - 1, "${PETSC_DIR}/share/petsc/amgx/AMG_CLASSICAL_AGGRESSIVE_L1_RT6.json");
    CHKERRQ(ierr);
    ierr = PetscStrreplace(PetscObjectComm((PetscObject)pc), amgx->filename, amgx->filename, PETSC_MAX_PATH_LEN);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
