#include "Parabolic2D.hpp"

void
Parabolic2D::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
    
    quadrature_face = std::make_unique<Quadrature<dim - 1>>(fe->degree + 1);
    
    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
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
Parabolic2D::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

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

        
          Vector<double> b(dim);
          advection_coefficient.vector_value(fe_values.quadrature_point(q), b);

          Tensor<1, dim> b_tensor;
          for (unsigned int d = 0; d < dim; ++d) {
            b_tensor[d] = b(d);  // Copy components from Vector to Tensor
          }


          // Evaluate coefficients on this quadrature node.
          const double diffusion_coefficient_loc = diffusion_coefficient.value(fe_values.quadrature_point(q));
          const double reaction_coefficient_loc = reaction_coefficient.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                cell_mass_matrix(i, j) += fe_values.shape_value(i, q)
                                          * fe_values.shape_value(j, q)
                                          / deltat
                                          * fe_values.JxW(q);
                
                //diffusion
                cell_stiffness_matrix(i, j) += diffusion_coefficient_loc
                                               * fe_values.shape_grad(i, q)
                                               * fe_values.shape_grad(j, q)
                                               * fe_values.JxW(q);
                
                //advection
                cell_stiffness_matrix(i, j) += scalar_product(b_tensor, fe_values.shape_grad(j,q))
                                               * fe_values.shape_value(i,q)
                                               * fe_values.JxW(q);
                
                //reaction
                cell_stiffness_matrix(i, j) += reaction_coefficient_loc
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
Parabolic2D::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();
  
  
  
  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points | update_JxW_values);
  
  
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

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

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =forcing_term.value(fe_values.quadrature_point(q));

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
      //Neumann boundary conditions
      if (cell->at_boundary())
      {

        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
        {

          if (cell->face(face_number)->at_boundary()){
            const bool atBoundary0 = cell->face(face_number)->boundary_id() == 0;
            const bool atBoundary1 = cell->face(face_number)->boundary_id() == 111;
            const bool atBoundary2 = cell->face(face_number)->boundary_id() == 211;
            const bool atBoundary3 = cell->face(face_number)->boundary_id() == 311;

            fe_face_values.reinit(cell, face_number);
            
            neumann_func.set_time(time);
            exact_solution.set_time(time);
            
            for (unsigned int q = 0; q < quadrature_face->size(); ++q){
              double neumann_value = 0;
              if(atBoundary0) neumann_value = -neumann_func.value(fe_face_values.quadrature_point(q), 0);
              if(atBoundary1) neumann_value = neumann_func.value(fe_face_values.quadrature_point(q), 0);
              if(atBoundary2) neumann_value = -neumann_func.value(fe_face_values.quadrature_point(q), 1);
              if(atBoundary3) neumann_value = neumann_func.value(fe_face_values.quadrature_point(q), 1);

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) += neumann_value
                              * fe_face_values.shape_value(i, q)
                              * fe_face_values.JxW(q);
                
              }
            }
          }
          
        }
      }
       */
      
      
      
      // Neumann boundary condition
      if (cell->at_boundary())
      {
        
        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
        {
          
          if (cell->face(face_number)->at_boundary() && (
                  cell->face(face_number)->boundary_id() == 1 ||
                  cell->face(face_number)->boundary_id() == 3
                  ))
          {
            fe_face_values.reinit(cell, face_number);
            
            
            for (unsigned int q = 0; q < n_q_face; ++q){
              
              //------------this section is only for proving correctness
              /*Vector<double> b_bound(dim);
              advection_coefficient.vector_value(fe_face_values.quadrature_point(q), b_bound);
              
              Tensor<1, dim> b_bound_tensor;
              for (unsigned int d = 0; d < dim; ++d) {
                b_bound_tensor[d] = b_bound(d);
              }
              
              cout << b_bound_tensor;
              const double neumann_value = diffusion_coefficient.value(fe_face_values.quadrature_point(q))
                                           * scalar_product(exact_solution.gradient(fe_face_values.quadrature_point(q)), fe_face_values.normal_vector(q));
              */
              //------------------------------------------------------------------------------------------
              
              //const double neumann_value = NeumannFunction.value(fe_face_values.quadrature_point(q));
              
              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) += 2.0

                               * fe_face_values.shape_value(i, q)
                               * fe_face_values.JxW(q);
              }
            }
          }
        }
      }
      
      

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  rhs_matrix.vmult_add(system_rhs, solution_owned);

  //Dirichlet boundary conditions
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    dirichlet_func.set_time(time);
    boundary_functions[0] = &dirichlet_func;
    boundary_functions[2] = &dirichlet_func;


    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, lhs_matrix, solution_owned, system_rhs, false);
  }
}

void
Parabolic2D::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);
  Eigen::IncompleteLUT preconditioner;
  preconditioner.initialize(lhs_matrix, preconditioner);

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " Solver iterations" << std::endl;

  solution = solution_owned;
}

void
Parabolic2D::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();


  std::string outName = "output_" + std::to_string(deltat) + "_";
  data_out.write_vtu_with_pvtu_record("./", outName.c_str(), time_step, MPI_COMM_WORLD, 2);

}

void
Parabolic2D::solve()
{
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_rhs(time);
      solve_time_step();
      output(time_step);
    }
}

double
Parabolic2D::compute_error(const VectorTools::NormType &norm_type)
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

  exact_solution.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}