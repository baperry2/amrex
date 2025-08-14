#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
      amrex::ParmParse pp;
      int grid_size = 128, max_box_size = 32;
      pp.query("grid_size", grid_size);
      pp.query("max_box_size", max_box_size);
      bool use_constant = false;
      pp.query("use_constant", use_constant);
      amrex::Real constant_value = 9.0;
      if (use_constant) {
	pp.query("constant_value", constant_value);
      }

      const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(grid_size - 1));
      const amrex::RealBox real_dom(amrex::RealVect(0.0).begin(), amrex::RealVect(1.0).begin());
      const int coord = 0;
      const amrex::Array<int,AMREX_SPACEDIM> is_per({AMREX_D_DECL(1,1,1)});
      amrex::Geometry geom(domain, real_dom, coord, is_per);
      amrex::BoxArray ba(domain);
      ba.maxSize(max_box_size);
      amrex::DistributionMapping dm{ba};
      const int num_grow = 0;
      const int num_comp = 2;
      amrex::MultiFab data(ba, dm, num_comp, num_grow);
      data.setVal(0.0,0,1); // set 1st component to 1
      data.setVal(1.0,1,1); // set 2nd component to 2

      const bool hard_coded_use_constant = false;
    amrex::Print() << "Starting test 1\n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(data, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto const& var_in_arr = data.const_array(mfi,0);
      auto const& var_out_arr = data.array(mfi,1);
      
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	  const amrex::Real set_val = hard_coded_use_constant ? constant_value : var_in_arr(i,j,k);
	  var_out_arr(i,j,k) = set_val;
        });
    }
    amrex::Print() << "Passed test 1, starting test 2 \n";
      
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(data, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto const& var_in_arr = data.const_array(mfi,0);
      auto const& var_out_arr = data.array(mfi,1);
      
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	  const amrex::Real set_val = use_constant ? constant_value : var_in_arr(i,j,k);
	  var_out_arr(i,j,k) = set_val;
        });
    }
    amrex::Print() << "Passed test 2 \n";
    }
    amrex::Finalize();
}
