#include "Poisson1D_DN.hpp"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  Poisson1D_DN problem_0(0);
  Poisson1D_DN problem_1(1);

  problem_0.setup();
  problem_1.setup();

  std::cout << "Setup completed" << std::endl;

  // Stopping criterion (same structure as exercise-01.cpp):
  // stop when ||u2^{k+1}-u2^k||_2 < tolerance_increment.
  const double       tolerance_increment = 1e-4;
  const unsigned int n_max_iter          = 100;

  double       solution_increment_norm = tolerance_increment + 1.0;
  unsigned int n_iter                  = 0;

  // Relaxation coefficient (1 = no relaxation).
  const double lambda = 0.25;

  while (n_iter < n_max_iter && solution_increment_norm > tolerance_increment)
    {
      auto solution_1_increment = problem_1.get_solution();

      // DN iteration:
      // - solve on Omega1 with Dirichlet interface (trace from Omega2)
      // - solve on Omega2 with Neumann interface (flux from Omega1)
      problem_0.assemble();
      problem_0.apply_interface_dirichlet(problem_1);
      problem_0.solve();

      problem_1.assemble();
      problem_1.apply_interface_neumann(problem_0);
      problem_1.solve();

      // Relaxation (applied on the Neumann subproblem, like in the lab code).
      problem_1.apply_relaxation(solution_1_increment, lambda);

      solution_1_increment -= problem_1.get_solution();
      solution_increment_norm = solution_1_increment.l2_norm();

      std::cout << "iteration " << n_iter
                << " - solution increment = " << solution_increment_norm
                << std::endl;

      problem_0.output(n_iter);
      problem_1.output(n_iter);

      ++n_iter;
    }

  return 0;
}
