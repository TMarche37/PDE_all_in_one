#include "Poisson1D.hpp"

void Poisson1D::setup()
{
    //Create the mesh
    GridGenerator::subdivided_hyper_cube(mesh,N_el,0.0,1.0,/*colorize*/true);
    std::cout<<"Number of nodes"<<mesh.n_active_cells() << std::endl;

    //Initialize the finite element space
    fe = std::make_unique<FE_SimplexP<dim>>(r);
    quadrature = std::make_unique<QGaussSimplex<dim>>(r+1);

    std::cout << "Dof per cell: "<< fe->dofs_per_cell << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    std::cout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

    //Initialize linear algebra
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
}

void Poisson1D::assemble()
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell; // n_loc
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    for(const auto &cell: dof_handler.active_cell_iterators()){
        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_rhs = 0.0;
        
        for(unsigned int q=0;q<n_q;++q){
            const auto mu_loc = mu(fe_values.quadrature_point(q));
            const auto f_loc = f(fe_values.quadrature_point(q));
            for(unsigned int i=0;i< dofs_per_cell;++i){
                for(unsigned int j=0;j< dofs_per_cell;++j){

                    cell_matrix(i,j) += mu_loc*fe_values.shape_grad(j,q)*fe_values.shape_grad(i,q)*fe_values.JxW(q);
                }
                cell_rhs(i) += f_loc * fe_values.shape_value(i,q) * fe_values.JxW(q);
            }
        }
        cell->get_dof_indices(dof_indices);

        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_rhs);
    }

    //Dirichlet boundary condition
    std::map<types::global_dof_index,double> boundary_values;

    Functions::ZeroFunction<dim> bc_function;
    
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    boundary_functions[0] = &bc_function;
    boundary_functions[1] = &bc_function;

    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, true);
}

void Poisson1D::solve()
{
    ReductionControl solver_control(/*maxiter=*/1000,/*tolerance=*/1.0e-16,/*reduce=*/1.0e-6);
    SolverCG<Vector<double>>solver(solver_control);

    solver.solve(system_matrix,solution,system_rhs,PreconditionIdentity());
    std::cout << "" << solver_control.last_step() << " CG iterations" <<std::endl;
}

void Poisson1D::output() const
{
    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, solution, "solution");

    data_out.build_patches();

    const std::string output_file_name="output-"+ std::to_string(N_el) + ".vtk";
    std::ofstream output_file(output_file_name);
    data_out.write_vtk(output_file);


}

double Poisson1D::compute_error(const VectorTools::NormType &norm_type, const Function<dim> &exact_solution) const
{
    const QGaussSimplex<dim> quadrature_error(r+2);

    Vector<double> error_per_cell(mesh.n_active_cells());
    VectorTools::integrate_difference(dof_handler, solution, exact_solution, error_per_cell, quadrature_error, norm_type);

    const double error = VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

    return error;
}