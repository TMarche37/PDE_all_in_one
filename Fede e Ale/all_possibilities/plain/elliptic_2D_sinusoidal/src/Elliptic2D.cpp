#include "Elliptic2D.hpp"

void
Elliptic2D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh from " << mesh_file_name << std::endl;

    // Read the mesh from file:
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);

    std::ifstream mesh_file(mesh_file_name);
    grid_in.read_msh(mesh_file);

    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    // Construct the finite element object. Notice that we use the FE_SimplexP
    // class here, that is suitable for triangular (or tetrahedral) meshes.
    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    // Construct the quadrature formula of the appopriate degree of exactness.
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;


    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    std::cout << "  Quadrature points per boundary cell = "
              << quadrature_boundary->size() << std::endl;

  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

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
Elliptic2D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();
  
  FEValues<dim> fe_values(
          *fe,
          *quadrature,
          update_values | update_gradients | update_quadrature_points |
          update_JxW_values);


  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                       update_quadrature_points |
                                       update_normal_vectors |
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
      
      Vector<double> b(dim);
      advection_coefficient.vector_value(fe_values.quadrature_point(q), b);

      Tensor<1, dim> b_tensor;
      for (unsigned int d = 0; d < dim; ++d) {
        b_tensor[d] = b[d];  // Copy components from Vector to Tensor
      }


      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {

          //diffusion term
          cell_matrix(i, j) += diffusion_coefficient.value(fe_values.quadrature_point(q)) // mu(x)
                               * fe_values.shape_grad(i, q)     // (I)
                               * fe_values.shape_grad(j, q)     // (II)
                               * fe_values.JxW(q);              // (III)


          //advection term
          cell_matrix(i,j) += -scalar_product(b_tensor, fe_values.shape_grad(i, q))
                              * fe_values.shape_value(j, q)
                              * fe_values.JxW(q);
          
          
          
          //reaction term
          cell_matrix(i, j) += reaction_coefficient.value(fe_values.quadrature_point(q))
                               * fe_values.shape_value(i, q)
                               * fe_values.shape_value(j, q)
                               * fe_values.JxW(q);
          }
        
        //rhs
        cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q))
                       * fe_values.shape_value(i, q)
                       * fe_values.JxW(q);


      }
    }


    // Neumann boundary condition
    if (cell->at_boundary())
    {

      for (unsigned int face_number = 0; face_number < cell->n_faces();
           ++face_number)
      {

        if (cell->face(face_number)->at_boundary()){
            const bool atBoundary0 = cell->face(face_number)->boundary_id() == 011;
            const bool atBoundary1 = cell->face(face_number)->boundary_id() == 1;
            const bool atBoundary2 = cell->face(face_number)->boundary_id() == 211;
            const bool atBoundary3 = cell->face(face_number)->boundary_id() == 3;

            fe_values_boundary.reinit(cell, face_number);
            
            for (unsigned int q = 0; q < quadrature_boundary->size(); ++q){
              double neumann_value = 0;
              if(atBoundary0) neumann_value = -neumann_func1.value(fe_values_boundary.quadrature_point(q));
              if(atBoundary1) neumann_value = neumann_func1.value(fe_values_boundary.quadrature_point(q));
              if(atBoundary2) neumann_value = -neumann_func3.value(fe_values_boundary.quadrature_point(q));
              if(atBoundary3) neumann_value = neumann_func3.value(fe_values_boundary.quadrature_point(q));

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) += neumann_value
                              * fe_values_boundary.shape_value(i, q)
                              * fe_values_boundary.JxW(q);
              }
            }
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
    boundary_functions[0] = &exact_solution;
    // boundary_functions[1] = &exact_solution;
    boundary_functions[2] = &exact_solution;
    // boundary_functions[3] = &exact_solution;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);


    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
Elliptic2D::solve()
{
  std::cout << "===============================================" << std::endl;

  SolverControl solver_control(10000, 1e-9);// * system_rhs.l2_norm());

  // If the preconditioner is not symmetric, we need to use the GMRES method.
  // SolverGMRES<Vector<double>> solver(solver_control);
  // SolverCG<Vector<double>> solver(solver_control);
  SolverBicgstab<Vector<double>> solver(solver_control);

  PreconditionSSOR preconditioner;
  preconditioner.initialize(system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));


  std::cout << "  Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "  " << solver_control.last_step() << " Solver iterations" << std::endl;
}



void
Elliptic2D::output() const
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;
  
  data_out.add_data_vector(dof_handler, solution, "solution");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string           output_file_name = "output-" + mesh_path.stem().string() + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}


double
Elliptic2D::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);


  const QGaussSimplex<dim> quadrature_error(r + 2);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}