# Runtime settings shared by the SLURM scripts.
# One thread per DNS rank; the agent spawns the DNS ranks itself (MPI_Comm_spawn).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMPI_MCA_rmaps_base_oversubscribe=1
# Spawned ranks would otherwise be pinned to the same cores as those of another job on the node.
export OMPI_MCA_hwloc_base_binding_policy=none
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p outputs
