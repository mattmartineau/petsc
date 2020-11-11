static char help[] = "Tests for mesh interpolation\n\n";

/* TODO
*/

#include <petscdmplex.h>

/* List of test meshes

Triangle
--------
Test 0:
Two triangles sharing a face

        4
      / | \
     /  |  \
    /   |   \
   2  0 | 1  5
    \   |   /
     \  |  /
      \ | /
        3

should become

        4
      / | \
     8  |  9
    /   |   \
   2  0 7 1  5
    \   |   /
     6  |  10
      \ | /
        3

Tetrahedron
-----------
Test 0:
Two tets sharing a face

 cell   5 _______    cell
 0    / | \      \       1
     /  |  \      \
    /   |   \      \
   2----|----4-----6
    \   |   /      /
     \  |  /     /
      \ | /      /
        3-------

should become

 cell   5 _______    cell
 0    / | \      \       1
     /  |  \      \
   17   |   18 13  22
   / 8 19 10 \      \
  /     |     \      \
 2---14-|------4--20--6
  \     |     /      /
   \ 9  | 7  /      /
   16   |   15 11  21
     \  |  /      /
      \ | /      /
        3-------


Quadrilateral
-------------
Test 0:
Two quads sharing a face

   5-------4-------7
   |       |       |
   |   0   |   1   |
   |       |       |
   2-------3-------6

should become

   5--10---4--14---7
   |       |       |
  11   0   9   1  13
   |       |       |
   2---8---3--12---6

Test 1:
A quad and a triangle sharing a face

   5-------4
   |       | \
   |   0   |  \
   |       | 1 \
   2-------3----6

should become

   5---9---4
   |       | \
  10   0   8  12
   |       | 1 \
   2---7---3-11-6

Hexahedron
----------
Test 0:
Two hexes sharing a face

cell   9-------------8-------------13 cell
0     /|            /|            /|     1
     / |   15      / |   21      / |
    /  |          /  |          /  |
   6-------------7-------------12  |
   |   |     18  |   |     24  |   |
   |   |         |   |         |   |
   |19 |         |17 |         |23 |
   |   |  16     |   |   22    |   |
   |   5---------|---4---------|---11
   |  /          |  /          |  /
   | /   14      | /    20     | /
   |/            |/            |/
   2-------------3-------------10

should become

cell   9-----31------8-----42------13 cell
0     /|            /|            /|     1
    32 |   15      30|   21      41|
    /  |          /  |          /  |
   6-----29------7-----40------12  |
   |   |     17  |   |     23  |   |
   |  35         |  36         |   44
   |19 |         |18 |         |24 |
  34   |  16    33   |   22   43   |
   |   5-----26--|---4-----37--|---11
   |  /          |  /          |  /
   | 25   14     | 27    20    | 38
   |/            |/            |/
   2-----28------3-----39------10

*/

typedef struct {
  DM        dm;
  PetscInt  testNum;                      /* Indicates the mesh to create */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  PetscBool useGenerator;                 /* Construct mesh with a mesh generator */
  PetscBool refCell;                      /* Construct reference cell by type */
  PetscBool femGeometry;                  /* Flag to compute FEM geometry */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->testNum      = 0;
  options->dim          = 2;
  options->cellSimplex  = PETSC_TRUE;
  options->useGenerator = PETSC_FALSE;
  options->refCell      = PETSC_FALSE;
  options->femGeometry  = PETSC_TRUE;
  options->filename[0]  = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-testnum", "The mesh to create", "ex7.c", options->testNum, &options->testNum, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex7.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex7.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex7.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ref_cell", "Create a reference cell", "ex7.c", options->refCell, &options->refCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fem_geometry", "Flag to compute FEM geometry", "ex7.c", options->femGeometry, &options->femGeometry, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex7.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_2D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {4, 2};
      PetscInt    coneSize[6]         = {3, 3, 0, 0, 0, 0};
      PetscInt    cones[6]            = {2, 3, 4,  5, 4, 3};
      PetscInt    coneOrientations[6] = {0, 0, 0,  0, 0, 0};
      PetscScalar vertexCoords[8]     = {-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5};
      PetscInt    markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_3D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[23]        = {4, 4, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {2, 4, 3, 5,  3, 4, 6, 5};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[15]    = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0,  1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
      PetscInt    markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscInt testNum, DM dm)
{
  PetscInt       depth = 1, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {6, 2};
      PetscInt    coneSize[8]         = {4, 4, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {2, 3, 4, 5,  3, 6, 7, 4};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
      PetscInt    markerPoints[12]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 6; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[7]         = {4, 3, 0, 0, 0, 0, 0};
      PetscInt    cones[7]            = {2, 3, 4, 5,  3, 6, 4};
      PetscInt    coneOrientations[7] = {0, 0, 0, 0,  0, 0, 0};
      PetscScalar vertexCoords[10]    = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, 0.5, 0.0};
      PetscInt    markerPoints[10]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 5; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateHex_3D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]         = {12, 2};
      PetscInt    coneSize[14]         = {8, 8, 0,0,0,0,0,0,0,0,0,0,0,0};
      PetscInt    cones[16]            = {2,5,4,3,6,7,8,9,  3,4,11,10,7,12,13,8};
      PetscInt    coneOrientations[16] = {0,0,0,0,0,0,0,0,  0,0, 0, 0,0, 0, 0,0};
      PetscScalar vertexCoords[36]     = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0,
                                          -0.5,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, -0.5,1.0,1.0,
                                           0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt    markerPoints[16]     = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 8; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckMesh(DM dm, AppCtx *user)
{
  PetscReal      detJ, J[9], refVol = 1.0;
  PetscReal      vol;
  PetscInt       dim, depth, d, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    refVol *= 2.0;
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    if (user->femGeometry) {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ);CHKERRQ(ierr);
      if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted, |J| = %g", c, (double) detJ);
      ierr = PetscInfo1(dm, "FEM Volume: %g\n", (double) detJ*refVol);CHKERRQ(ierr);
    }
    if (depth > 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      if (vol <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted, vol = %g", c, (double) vol);
      ierr = PetscInfo1(dm, "FVM Volume: %g\n", (double) vol);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CompareCones(DM dm, DM idm)
{
  PetscInt       cStart, cEnd, c, vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt *cone;
    PetscInt       *points = NULL, numPoints, p, numVertices = 0, coneSize;

    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(idm, c, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints*2; p += 2) {
      const PetscInt point = points[p];
      if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
    }
    if (numVertices != coneSize) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In cell %d, cone size %d != %d vertices in closure", c, coneSize, numVertices);
    for (v = 0; v < numVertices; ++v) {
      if (cone[v] != points[v]) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "In cell %d, cone point %d is %d != %d vertex in closure", c, v, cone[v], points[v]);
    }
    ierr = DMPlexRestoreTransitiveClosure(idm, c, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, PetscInt testNum, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscBool      cellSimplex  = user->cellSimplex;
  PetscBool      useGenerator = user->useGenerator;
  const char    *filename     = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_FALSE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  } else if (useGenerator) {
    ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, NULL, NULL, NULL, NULL, PETSC_FALSE, dm);CHKERRQ(ierr);
  } else if (user->refCell) {
    ierr = DMPlexCreateReferenceCellByType(comm, DM_POLYTOPE_UNKNOWN, dm);CHKERRQ(ierr);
  } else {
    ierr = DMCreate(comm, dm);CHKERRQ(ierr);
    ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      if (cellSimplex) {
        ierr = CreateSimplex_2D(comm, *dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, testNum, *dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (cellSimplex) {
        ierr = CreateSimplex_3D(comm, *dm);CHKERRQ(ierr);
      } else {
        ierr = CreateHex_3D(comm, *dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
    }
  }
  {
    DMPlexInterpolatedFlag interpolated;

    ierr = CheckMesh(*dm, user);CHKERRQ(ierr);
    ierr = DMPlexIsInterpolated(*dm, &interpolated);CHKERRQ(ierr);
    if (interpolated == DMPLEX_INTERPOLATED_FULL) {
      DM udm;

      ierr = DMPlexUninterpolate(*dm, &udm);CHKERRQ(ierr);
      ierr = CompareCones(udm, *dm);CHKERRQ(ierr);
      ierr = DMDestroy(&udm);CHKERRQ(ierr);
    } else {
      DM idm;

      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = CompareCones(*dm, idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = idm;
    }
  }
  {
    DM               distributedMesh = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMViewFromOptions(distributedMesh, NULL, "-dm_view");CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, user.testNum, &user, &user.dm);CHKERRQ(ierr);
  ierr = CheckMesh(user.dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # Two cell test meshes 0-7
  test:
    suffix: 0
    args: -dim 2 -dm_view ascii::ascii_info_detail
  test:
    suffix: 1
    nsize: 2
    args: -petscpartitioner_type simple -dim 2 -dm_view ascii::ascii_info_detail
  test:
    suffix: 2
    args: -dim 2 -cell_simplex 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 3
    nsize: 2
    args: -petscpartitioner_type simple -dim 2 -cell_simplex 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 4
    args: -dim 3 -dm_view ascii::ascii_info_detail
  test:
    suffix: 5
    nsize: 2
    args: -petscpartitioner_type simple -dim 3 -dm_view ascii::ascii_info_detail
  test:
    suffix: 6
    args: -dim 3 -cell_simplex 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 7
    nsize: 2
    args: -petscpartitioner_type simple -dim 3 -cell_simplex 0 -dm_view ascii::ascii_info_detail
  # 2D Hybrid Mesh 8
  test:
    suffix: 8
    args: -dim 2 -cell_simplex 0 -testnum 1 -dm_view ascii::ascii_info_detail
  # Reference cells
  test:
    suffix: 12
    args: -ref_cell -dm_plex_ref_type pyramid -fem_geometry 0 -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces
  # TetGen meshes 9-10
  test:
    suffix: 9
    requires: triangle
    args: -dim 2 -use_generator -dm_view ascii::ascii_info_detail
  test:
    suffix: 10
    requires: ctetgen
    args: -dim 3 -use_generator -dm_view ascii::ascii_info_detail
  # Cubit meshes 11
  test:
    suffix: 11
    requires: exodusii
    args: -dim 3 -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -dm_view ascii::ascii_info_detail

TEST*/
