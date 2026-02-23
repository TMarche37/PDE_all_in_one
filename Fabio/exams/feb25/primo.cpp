#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "Poisson1D.hpp"

static constexpr unsigned int dim = Poisson1D::dim;


class ExactSolution : public Function<dim>
{
    public:
        ExactSolution() = default;

        // Evaluation.
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            // Points 3 and 4.
            return p[0]-p[0]*p[0]; // x*(1-x)

            // Point 5.
            // if (p[0] < 0.5)
            //   return A * p[0];
            // else
            //   return A * p[0] + 4.0 / 15.0 * std::pow(p[0] - 0.5, 2.5);
        }
        // Gradient evaluation.
        // deal.II requires this method to return a Tensor (not a double), i.e. a
        // dim-dimensional vector. In our case, dim = 1, so that the Tensor will in
        // practice contain a single number. Nonetheless, we need to return an
        // object of type Tensor.
        virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            Tensor<1, dim> result;

            // Points 3 and 4.
            result[0] = 1.0 - 2.0 * p[0]; // 1 - 2x derivative of x(1-x)

            // Point 5.
            // if (p[0] < 0.5)
            //   result[0] = A;
            // else
            //   result[0] = A + 2.0 / 3.0 * std::pow(p[0] - 0.5, 1.5);

            return result;
        }

};


//Main function
int main(int /*argc*/, char* /*argv*/[])
{

    constexpr unsigned int dim = Poisson1D::dim;

    const std::vector<unsigned int> N_el = {10,20,40,80,160};

    std::ofstream convergence_file("convergence.csv");
    convergence_file << "h,eL2,eH1" << std::endl;

    ConvergenceTable table;

    const unsigned int r=2;

    const auto mu=[](const Point<dim> & /*p*/){
        return 1.0;
    };

    const auto b=[](const Point<dim> & p){
        return -p[0];
    };

    const auto sigma=[](const Point<dim> & /*p*/){
        return 1.0;
    };

    const auto f =[](const Point<dim> &p){
        return p[0]*p[0] + 2.0; // x^2 + 2
    };
    for (const unsigned int &j : N_el)
    {
        ExactSolution exact_solution;
        Poisson1D problem(j,r,mu,b,sigma,f);

        problem.setup();
        problem.assemble();
        problem.solve();
        problem.output();


        const double h = 1.0 / j;

        const double error_L2 = problem.compute_error(VectorTools::L2_norm, exact_solution);
        const double error_H1 = problem.compute_error(VectorTools::H1_norm, exact_solution);

        table.add_value("h", h);
        table.add_value("L2", error_L2);
        table.add_value("H1", error_H1);

        convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;

    }
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

    table.set_scientific("L2", true);
    table.set_scientific("H1", true);

    table.write_text(std::cout);


    return 0;
}

