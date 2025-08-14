AMREX_HOME = ../../../

DEBUG	= FALSE

DIM	= 3

COMP    = intel-llvm

USE_CUDA  = FALSE
USE_ACC   = FALSE
USE_OMP_OFFLOAD = FALSE

USE_SYCL  = TRUE
USE_MPI   = TRUE
USE_OMP   = FALSE

TINY_PROFILE = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
