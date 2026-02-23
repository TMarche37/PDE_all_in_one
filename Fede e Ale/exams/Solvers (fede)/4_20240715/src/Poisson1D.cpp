#include "Poisson1D.hpp"

void
Poisson1D::setup() {
  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube(mesh, N + 1, 0.0, 1.0, true);
    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique < FE_Q < dim >> (r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique < QGauss < dim >> (r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;


    quadrature_boundary = std::make_unique < QGauss < dim - 1 >> (r + 1);

    pcout << "  Quadrature points per boundary cell = "
          << quadrature_boundary->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
Poisson1D::assemble_matrices()
{
  cout << "===============================================" << std::endl;
  cout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);


  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {

    if (!cell->is_locally_owned())
      continue;


    fe_values.reinit(cell);

    cell_mass_matrix      = 0.0;
    cell_stiffness_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) /
                                    deltat *
                                    fe_values.JxW(q);
          //diffusion
          cell_stiffness_matrix(i, j) += diffusion_coefficient.value(fe_values.quadrature_point(q))
                                         * fe_values.shape_grad(i, q)     // (I)
                                         * fe_values.shape_grad(j, q)     // (II)
                                         * fe_values.JxW(q);              // (III)

          //pcout << fe_values.shape_grad(i,q)[0]<<std::endl;
          //advection
          cell_stiffness_matrix(i,j) += advection_coefficient.value(fe_values.quadrature_point(q))
                                        * fe_values.shape_value(i,q)
                                        * fe_values.shape_grad(j,q)[0]
                                        * fe_values.JxW(q);

          //reaction
          cell_stiffness_matrix(i,j) += -reaction_coefficient.value(fe_values.quadrature_point(q)) // sigma(x)
                                        * fe_values.shape_value(i,q)
                                        * fe_values.shape_value(j,q)
                                        * fe_values.JxW(q);
        }
      }
    }

    cell->get_dof_indices(dof_indices);

    mass_matrix.add(dof_indices, cell_mass_matrix);
    stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);

}


void
Poisson1D::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                          update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // We need to compute the forcing term at the current time (tn+1) and
      // at the old time (tn). deal.II Functions can be computed at a
      // specific time by calling their set_time method.

      // Compute f(tn+1)
      forcing_term.set_time(time);
      const double f_new_loc = forcing_term.value(fe_values.quadrature_point(q));

      // Compute f(tn)
      forcing_term.set_time(time - deltat);
      const double f_old_loc = forcing_term.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                       fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    /*
    if (cell->at_boundary())
    {
      // ...we loop over its edges (referred to as faces in the deal.II
      // jargon).
      for (unsigned int face_number = 0; face_number < cell->n_faces();
           ++face_number)
      {
        // If current face lies on the boundary, and its boundary ID (or
        // tag) is that of one of the Neumann boundaries, we assemble the
        // boundary integral.
        if (cell->face(face_number)->at_boundary() &&
            (cell->face(face_number)->boundary_id() == 1))
        {
          fe_values_boundary.reinit(cell, face_number);

          beta.set_time(time);
          for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += beta.value(fe_values_boundary.quadrature_point(q)) * // h(xq)
                             fe_values_boundary.shape_value(i, q) *      // v(xq)
                             fe_values_boundary.JxW(q);                  // Jq wq
        }
      }
    }*/

    cell->get_dof_indices(dof_indices);
    system_rhs.add(dof_indices, cell_rhs);
  }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  rhs_matrix.vmult_add(system_rhs, solution_owned);

  // We apply boundary conditions to the algebraic system.
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    Functions::ZeroFunction<dim> zeroFunction(dim);
    boundary_functions[0] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
            boundary_values, lhs_matrix, solution_owned, system_rhs, false);
  }
}



void
Poisson1D::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
          lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  cout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}


void
Poisson1D::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
          "./", "output", time_step, MPI_COMM_WORLD, 3);
}


void
Poisson1D::solve()
{
  assemble_matrices();

  cout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    cout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    cout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T)
  {

    time += deltat;
    ++time_step;



    cout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
          << time << ":" << std::flush;

    assemble_rhs(time);
    solve_time_step();
    output(time_step);
  }
}





double
Poisson1D::compute_error(const VectorTools::NormType &norm_type)
{
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGauss<dim> quadrature_error = QGauss<dim>(r + 2);

  exact_solution.set_time(time);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
          VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}

















//------------------------




/*


void
Poisson1D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements.
  FEValues<dim> fe_values(
          *fe,
          *quadrature,
          // Here we specify what quantities we need FEValues to compute on
          // quadrature points. For our test, we need:
          // - the values of shape functions (update_values);
          // - the derivative of shape functions (update_gradients);
          // - the position of quadrature points (update_quadrature_points);
          // - the product J_c(x_q)*w_q (update_JxW_values).
          update_values | update_gradients | update_quadrature_points |
          update_JxW_values);

  // Local matrix and right-hand side vector. We will overwrite them for
  // each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the
  // current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    // Reinitialize the FEValues object on current element. This
    // precomputes all the quantities we requested when constructing
    // FEValues (see the update_* flags above) for all quadrature nodes of
    // the current cell.
    fe_values.reinit(cell);

    // We reset the cell matrix and vector (discarding any leftovers from
    // previous element).
    cell_matrix = 0.0;
    cell_rhs    = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // Here we assemble the local contribution for current cell and
      // current quadrature point, filling the local matrix and vector.

      // Here we iterate over *local* DoF indices.
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // FEValues::shape_grad(i, q) returns the gradient of the i-th
          // basis function at the q-th quadrature node, already mapped
          // on the physical element: we don't have to deal with the
          // mapping, it's all hidden inside FEValues.


          //diffusion
          cell_matrix(i, j) += * fe_values.shape_grad(i, q)     // (I)
                               * fe_values.shape_grad(j, q)     // (II)
                               * fe_values.JxW(q);              // (III)

          //reaction
          cell_matrix(i,j) += reaction_coefficient.value(fe_values.quadrature_point(q)) // sigma(x)
                              * fe_values.shape_value(i,q)
                              * fe_values.shape_value(i,q)
                              * fe_values.JxW(q);
        }

        //cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
        //               fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    // At this point the local matrix and vector are constructed: we need
    // to sum them into the global matrix and vector. To this end, we need
    // to retrieve the global indices of the DoFs of current cell.
    cell->get_dof_indices(dof_indices);

    // Then, we add the local matrix and vector into the corresponding
    // positions of the global matrix and vector.
    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }

  // Boundary conditions.
  {
    // We construct a map that stores, for each DoF corresponding to a Dirichlet
    // condition, the corresponding value. E.g., if the Dirichlet condition is
    // u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // This object represents our boundary data as a real-valued function (that
    // always evaluates to zero).
    Functions::ZeroFunction<dim> bc_function;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &bc_function;
    boundary_functions[1] = &bc_function;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary conditions.
    // This replaces the equations for the boundary DoFs with the corresponding
    // u_i = 0 equations.
    MatrixTools::apply_boundary_values(
            boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Poisson1D::solve()
{
  std::cout << "===============================================" << std::endl;

  // Here we specify the maximum number of iterations of the iterative solver,
  // and its tolerance.
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  // Since the system matrix is symmetric and positive definite, we solve the
  // system using the conjugate gradient method.
  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  // We don't use any preconditioner for now, so we pass the identity matrix as
  // preconditioner.
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void
Poisson1D::output() const
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Once all vectors have been inserted, call build_patches to finalize the
  // DataOut object, preparing it for writing to file.
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  const std::string output_file_name =
          "output-" + std::to_string(N + 1) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}


 */