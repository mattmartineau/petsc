/*
  This file implements a PC interface to the AMGx CUDA GPU solver library
*/
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

#include "cuda_runtime.h"

// AmgX
#include <amgx_c.h>

typedef struct _PC_AMGX
{
    AMGX_solver_handle AmgXsolver;
    AMGX_matrix_handle AmgXA;
    AMGX_vector_handle AmgXP;
    AMGX_vector_handle AmgXRHS;
    AMGX_config_handle cfg;
    MPI_Comm comm;
    AMGX_resources_handle rsrc;
    void *lib_handle;
    char filename[PETSC_MAX_PATH_LEN];
} PC_AMGX;
static PetscInt s_count = 0;

/* ----------------------------------------------------------------------------- */
PetscErrorCode PCReset_AMGX(PC pc)
{
    PC_AMGX *amgx = (PC_AMGX *)pc->data;

    PetscFunctionBegin;
    AMGX_solver_destroy(amgx->AmgXsolver);
    AMGX_matrix_destroy(amgx->AmgXA);
    AMGX_vector_destroy(amgx->AmgXP);
    AMGX_vector_destroy(amgx->AmgXRHS);
    PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_AMGX(PC pc)
{
    PetscErrorCode ierr;
    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    cudaError_t err = cudaSuccess;

    PetscFunctionBegin;
    ierr = PCReset(pc);
    CHKERRQ(ierr);
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
        ierr = MPI_Comm_free(&amgx->comm);
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
    ierr = PetscFree(amgx); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_AMGX(PC pc)
{
    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    PetscErrorCode ierr;
    cudaError_t err = cudaSuccess;
    Mat Pmat = pc->pmat;
    PetscInt nGlobalRows, nLocalRows, rawN, bs;
    int nranks, rank, nnz;
    PetscBool done;
    MPI_Comm wcomm = amgx->comm;

    PetscFunctionBegin;
    if (!pc->setupcalled)
    {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx->cfg, amgx->filename));

        /* switch on internal error handling (no need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx->cfg, "exception_handling=1"));

        /* create an AmgX resource object, only the first instance is in charge */

        // XXX Unless I missed something outside of this file, must change for multi GPU I would expect
        int devID, devCount;
        err = cudaGetDevice(&devID);
        if (err != cudaSuccess)
        {
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "error in cudaGetDevice %s", cudaGetErrorString(err));
        }
        err = cudaGetDeviceCount(&devCount);
        if (err != cudaSuccess)
        {
            SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "error in cudaGetDeviceCount %s", cudaGetErrorString(err));
        }

        err = AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &devID);
        err = AMGX_matrix_create(&amgx->AmgXA, amgx->rsrc, AMGX_mode_dDDI);
        err = AMGX_vector_create(&amgx->AmgXP, amgx->rsrc, AMGX_mode_dDDI);
        err = AMGX_vector_create(&amgx->AmgXRHS, amgx->rsrc, AMGX_mode_dDDI);
        err = AMGX_solver_create(&amgx->AmgXsolver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg);
    }
    else
    {
        // XXX Need to double check, but assuming "setupcalled" will be true after the first invocation,
        // we can simply re-setup / replace coefficients in future. How do we check if the
        // sparsity pattern changed though...?
    }

    /* upload matrix */
    MPI_Comm_size(wcomm, &nranks);
    MPI_Comm_rank(wcomm, &rank);

    ierr = MatGetSize(Pmat, &nGlobalRows, NULL); CHKERRQ(ierr);

    // XXX This isn't a requirement on AmgX, rather this integration doesn't yet handle 64bit indices correctly
    if (nGlobalRows >= 2147483648)
    {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMGx only supports 32 bit ints. N = %D", nGlobalRows);
    }

    ierr = MatGetLocalSize(Pmat, &nLocalRows, NULL); CHKERRQ(ierr);
    ierr = MatGetBlockSize(Pmat, &bs); CHKERRQ(ierr);

    {
        /* get offsets */
        PetscInt *rows;
        PetscInt *cols;
        PetscScalar *data;
        AMGX_distribution_handle dist;

        /* to calculate the partition offsets and pass those into the API call instead of creating a full partition vector. */
        PetscInt *partition_offsets;
        int nglobal32, i, n32 = nLocalRows, bs32 = bs;

        /* get raw matrix data */
        {
            Mat localA = 0;
            const PetscInt *rawCol, *rawRow;
            PetscScalar *rawData;

            /* get local matrix from redistributed matrix */
            if (nranks == 1)
            {
                localA = Pmat;
            }
            else
            {
                ierr = MatMPIAIJGetLocalMat(Pmat, MAT_INITIAL_MATRIX, &localA); CHKERRQ(ierr);
            }

            ierr = MatGetRowIJ(localA, 0, PETSC_FALSE, PETSC_FALSE, &rawN, &rawRow, &rawCol, &done); CHKERRQ(ierr);
            if(!done)
            {
                printf("MatGetRowIJ did not work\n");
            }
            if (rawN != nLocalRows)
            {
                SETERRQ2(wcomm, PETSC_ERR_PLIB, "rawN != nLocalRows %D %D\n", rawN, nLocalRows);
            }

            ierr = MatSeqAIJGetArray(localA, &rawData); CHKERRQ(ierr);

            /* copy values to STL vector. Note: there is an implicit conversion from */
            /* PetscInt to int64_t for the column vector */
            nnz = rawRow[nLocalRows];

            ierr = PetscMalloc1(nnz, &cols); CHKERRQ(ierr);
            ierr = PetscMalloc1(nnz, &data); CHKERRQ(ierr);
            ierr = PetscMalloc1(nLocalRows+1, &rows); CHKERRQ(ierr);

            // XXX Unless I am forgetting some technical limitation - I don't think this is necessary or beneficial.
            // We can surely instead just pass the data into AmgX and allow it to make the copies for us.
            // Otherwise we perform an additional unnecessary copy.
            for (i = 0; i < nnz; ++i)
            {
                cols[i] = rawCol[i];
                data[i] = rawData[i];
            }
            for (i = 0; i < nLocalRows + 1; ++i)
            {
                rows[i] = rawRow[i];
            }

            ierr = MatRestoreRowIJ(localA, 0, PETSC_FALSE, PETSC_FALSE, &rawN, &rawRow, &rawCol, &done); CHKERRQ(ierr);
            ierr = MatSeqAIJRestoreArray(localA, &rawData); CHKERRQ(ierr);

            if (localA != Pmat)
            {
                ierr = MatDestroy(&localA); CHKERRQ(ierr);
            }
        }

        /* pin the memory to improve performance
            WARNING: Even though, internal error handling has been requested,
            AMGX_SAFE_CALL needs to be used on this system call.
            It is an exception to the general rule. */
        /* AMGX_SAFE_CALL(AMGX_pin_memory(cols, nnz * sizeof(int64_t))); */
        /* AMGX_SAFE_CALL(AMGX_pin_memory(rows, (n32 + 1)*sizeof(int))); */
        /* AMGX_SAFE_CALL(AMGX_pin_memory(data, nnz * sizeof(PetscScalar))); */ /* check that this has bs^2 */
        /* get offsets and upload */

        ierr = PetscMalloc1(nranks + 1, &partition_offsets); CHKERRQ(ierr);

        partition_offsets[0] = 0; /* could use PetscLayoutGetRanges */

        ierr = MPI_Allgather(&nLocalRows, sizeof(PetscInt), MPI_BYTE, &partition_offsets[1], sizeof(PetscInt), MPI_BYTE, wcomm); CHKERRQ(ierr);

        for (i = 1; i <= nranks; i++)
        {
            partition_offsets[i] += partition_offsets[i - 1];
        }

        nglobal32 = partition_offsets[nranks]; // last element always has global number of rows

        /* upload - this takes an int for nglobal (does it work for large sysetms ??) */
        int petsc32 = (sizeof(PetscInt) == 4);
        AMGX_distribution_create(&dist, amgx->cfg);
        AMGX_distribution_set_32bit_colindices(dist, petsc32);
        AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets);
        AMGX_matrix_upload_distributed(amgx->AmgXA, nglobal32, n32, nnz, bs32, bs32, rows, cols, data, NULL, dist);
        AMGX_distribution_destroy(dist);

        ierr = PetscFree(partition_offsets); CHKERRQ(ierr);
        ierr = PetscFree(cols); CHKERRQ(ierr);
        ierr = PetscFree(data); CHKERRQ(ierr);
        ierr = PetscFree(rows); CHKERRQ(ierr);

        //ierr = PetscFree(partition_offsets); CHKERRQ(ierr);
        //ierr = PetscFree3(cols, data, rows); CHKERRQ(ierr);
    }

    /* bind the matrix A to the solver */
    ierr = MPI_Barrier(wcomm); CHKERRQ(ierr);

    AMGX_solver_setup(amgx->AmgXsolver, amgx->AmgXA);

    /* connect (bind) vectors to the matrix */
    AMGX_vector_bind(amgx->AmgXP, amgx->AmgXA);
    AMGX_vector_bind(amgx->AmgXRHS, amgx->AmgXA);

    PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_AMGX(PC pc, Vec b, Vec x)
{
    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    PetscErrorCode ierr;
    PetscScalar *unks;
    const PetscScalar *rhs;
    PetscInt n;
    AMGX_SOLVE_STATUS status;

    PetscFunctionBegin;

    ierr = VecGetLocalSize(x, &n); CHKERRQ(ierr);
    ierr = VecGetArray(x, &unks); CHKERRQ(ierr);
    ierr = VecGetArrayRead(b, &rhs); CHKERRQ(ierr);

    AMGX_vector_upload(amgx->AmgXP, n, 1, unks);
    AMGX_vector_upload(amgx->AmgXRHS, n, 1, rhs);

    ierr = MPI_Barrier(amgx->comm); CHKERRQ(ierr);

    AMGX_solver_solve(amgx->AmgXsolver, amgx->AmgXRHS, amgx->AmgXP);
    AMGX_solver_get_status(amgx->AmgXsolver, &status);

    if (status == AMGX_SOLVE_FAILED)
    {
        SETERRQ1(amgx->comm, PETSC_ERR_CONV_FAILED,
            "AmgX solver failed to solve the system! "
            "The error code is %d.\n", status);
    }

    AMGX_vector_download(amgx->AmgXP, unks);

    ierr = VecRestoreArray(x, &unks); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(b, &rhs); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_AMGX(PetscOptionItems *PetscOptionsObject, PC pc)
{
    PC_AMGX *amgx = (PC_AMGX *)pc->data;
    PetscErrorCode ierr;
    PetscBool exists;

    PetscFunctionBegin;
    ierr = PetscOptionsHead(PetscOptionsObject, "AMGX options"); CHKERRQ(ierr);
    ierr = PetscOptionsString("-pc_amgx_json", "AMGX parameter file (json)", "amgx.c", amgx->filename, amgx->filename, PETSC_MAX_PATH_LEN, NULL); CHKERRQ(ierr);
    ierr = PetscStrreplace(PetscObjectComm((PetscObject)pc), amgx->filename, amgx->filename, PETSC_MAX_PATH_LEN); CHKERRQ(ierr);
    ierr = PetscTestFile(amgx->filename, 'r', &exists); CHKERRQ(ierr);

    if (!exists)
    {
        printf("Parameter -pc_amgx_json incorrect.\n");

        /* try to add prefix */
        char str[PETSC_MAX_PATH_LEN];
        ierr = PetscSNPrintf(str, PETSC_MAX_PATH_LEN - 1, "${PETSC_DIR}/share/petsc/amgx/%s", amgx->filename); CHKERRQ(ierr);
        ierr = PetscStrreplace(PetscObjectComm((PetscObject)pc), str, amgx->filename, PETSC_MAX_PATH_LEN); CHKERRQ(ierr);
        ierr = PetscTestFile(amgx->filename, 'r', &exists); CHKERRQ(ierr);

        if (!exists)
        {
            SETERRQ1(PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "input file not found (%s)", amgx->filename);
        }
    }
    else
    {
        printf("As per -pc_amgx_json, found parameter file at %s.\n", amgx->filename);
    }
    ierr = PetscOptionsTail(); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode PCView_AMGX(PC pc, PetscViewer viewer)
{
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
    PetscErrorCode ierr;
    PC_AMGX *amgx;

    PetscFunctionBegin;
    ierr = PetscNewLog(pc, &amgx);
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
    { /* can put this in a PCAMGXFinalizePackage method */
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
        ierr = MPI_Comm_dup(comm_in, &amgx->comm); CHKERRQ(ierr);
    }
    /* set a default path/filename, use -pc_amgx_json to set at runtime */
    ierr = PetscSNPrintf(amgx->filename, PETSC_MAX_PATH_LEN - 1, "${PETSC_DIR}/share/petsc/amgx/AMG_CLASSICAL_AGGRESSIVE_L1_RT6.json"); CHKERRQ(ierr);
    ierr = PetscStrreplace(PetscObjectComm((PetscObject)pc), amgx->filename, amgx->filename, PETSC_MAX_PATH_LEN); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
