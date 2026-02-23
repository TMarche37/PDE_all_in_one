#include "Elliptic1D.hpp"

void
Elliptic1D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube(mesh, N + 1, 0.0, 1.0, true);
    std::cout << "  Number of elements = " << mesh.n_active_cells() << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    // Finite elements in one dimension are obtained with the FE_Q class
    // (which for higher dimensions represents hexahedral finite elements). In
    // higher dimensions, we must use FE_Q for hexahedral elements or
    // FE_SimplexP for tetrahedral elements. They are both derived from
    // FiniteElement, so that the code is generic.
    fe = std::make_unique<FE_Q<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;


    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
    
    
    quadrature_boundary = std::make_unique < QGauss < dim - 1 >> (r + 1);
    
    std::cout << "  Quadrature points per boundary cell = " << quadrature_boundary->size() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);


    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;


    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix
    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Elliptic1D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  
  
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                       update_quadrature_points |
                                       update_JxW_values);


  
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);


  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
    
      fe_values.reinit(cell);

      
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            
            cell_matrix(i, j) += diffusion_coefficient.value(fe_values.quadrature_point(q)) // mu(x)
                                 * fe_values.shape_grad(i, q)     // (I)
                                 * fe_values.shape_grad(j, q)     // (II)
                                 * fe_values.JxW(q);              // (III)
                                 
     
                                 
            cell_matrix(i, j) += advection_coefficient.value(fe_values.quadrature_point(q))
                                 * fe_values.shape_value(i, q)
                                 * fe_values.shape_grad(j, q)[0]
                                 * fe_values.JxW(q);
            
            
            cell_matrix(i, j) += reaction_coefficient.value(fe_values.quadrature_point(q))
                                 * fe_values.shape_value(i, q)
                                 * fe_values.shape_value(j, q)
                                 * fe_values.JxW(q);
            
            
          }

        cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
                       fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
      }
      
      
      // Neumann Boundary conditions
      if (cell->at_boundary())
      {

        for (unsigned int face_number = 0; face_number < cell->n_faces();
             ++face_number)
        {
          
          if (cell->face(face_number)->at_boundary() &&
              (cell->face(face_number)->boundary_id() == 1))
          {
            fe_values_boundary.reinit(cell, face_number);
            
            for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += neumann_func.value(fe_values_boundary.quadrature_point(q))
                               * fe_values_boundary.shape_value(i, q)
                               * fe_values_boundary.JxW(q);
          }
        }
      }


      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Dirichlet Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
  
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &dirichlet_func;
    
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);
    
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Elliptic1D::solve()
{
  std::cout << "===============================================" << std::endl;
  
  SolverControl solver_control(10000, 1e-8);//* system_rhs.l2_norm());
  
  //SolverCG<Vector<double>> solver(solver_control);
  // SolverGMRES<Vector<double>> solver(solver_control);
  SolverBicgstab<Vector<double>> solver(solver_control);

  // PreconditionIdentity preconditioner;
  PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);

  // PreconditionSOR preconditioner;
  // preconditioner.initialize(system_matrix, PreconditionSOR<SparseMatrix<double>>::AdditionalData(1.0));

  std::cout << "  Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution, system_rhs,  preconditioner);
  std::cout << "  " << solver_control.last_step() << " Solver iterations" << std::endl;
}

void
Elliptic1D::output() const
{
  std::cout << "===============================================" << std::endl;

  DataOut<dim> data_out;
  
  data_out.add_data_vector(dof_handler, solution, "solution");


  data_out.build_patches();
  
  const std::string output_file_name = "output-" + std::to_string(N + 1) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}

double
Elliptic1D::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_Q<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  const QGauss<dim> quadrature_error = QGauss<dim>(r + 2);

  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error = VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}