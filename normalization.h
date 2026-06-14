#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
#include <functional>

namespace nn {

    // Abstract Base Class for all Normalization techniques
    class Normalization {
    public:
        virtual ~Normalization() = default;
        
        virtual std::string name() const = 0;
        virtual void initialize(int dim) = 0;
        virtual std::vector<double> forward(const std::vector<double>& x) = 0;
        virtual std::vector<double> backward(const std::vector<double>& grad_output) = 0;

        // Expose state vectors so the Model can save/load them
        virtual bool has_params() const = 0;
        virtual std::vector<double>& get_gamma() = 0;
        virtual std::vector<double>& get_beta() = 0;
        virtual std::vector<double>& get_dgamma() = 0;
        virtual std::vector<double>& get_dbeta() = 0;
    };

    // Layer Normalization (using Lambda math as requested)
    class LayerNorm : public Normalization {
    private:
        int N;
        double epsilon;
        std::vector<double> cache_x_hat;
        double cache_var;
        
        // Learnable scaling and shifting matrices/vectors
        std::vector<double> gamma;
        std::vector<double> beta;
        std::vector<double> dgamma;
        std::vector<double> dbeta;

    public:
        LayerNorm(double eps = 1e-5) : N(0), epsilon(eps) {}

        void initialize(int dim) override {
            N = dim;
            gamma.assign(N, 1.0);  // Default scale is 1
            beta.assign(N, 0.0);   // Default shift is 0
            dgamma.assign(N, 0.0);
            dbeta.assign(N, 0.0);
            cache_x_hat.assign(N, 0.0);
        }

        std::vector<double> forward(const std::vector<double>& x) override {
            if (N == 0) initialize(x.size());
            std::vector<double> out(N, 0.0);

            // Math via Lambdas
            auto calc_mean = [](const std::vector<double>& v) {
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            };
            
            double mean = calc_mean(x);

            auto calc_var = [&mean](const std::vector<double>& v) {
                double sum_sq = 0.0;
                std::for_each(v.begin(), v.end(), [&sum_sq, mean](double val) {
                    sum_sq += (val - mean) * (val - mean);
                });
                return sum_sq / v.size();
            };

            cache_var = calc_var(x);
            double std_dev = std::sqrt(cache_var + epsilon);

            // Normalize, Scale, and Shift via lambda
            for (int i = 0; i < N; ++i) {
                cache_x_hat[i] = (x[i] - mean) / std_dev;
                out[i] = gamma[i] * cache_x_hat[i] + beta[i];
            }

            return out;
        }

        std::vector<double> backward(const std::vector<double>& grad_output) override {
            std::vector<double> grad_input(N, 0.0);
            double std_dev_inv = 1.0 / std::sqrt(cache_var + epsilon);

            // Calculate gradients for gamma and beta
            for (int i = 0; i < N; ++i) {
                dgamma[i] = grad_output[i] * cache_x_hat[i];
                dbeta[i] = grad_output[i];
            }

            // Math via Lambdas for input gradients
            double sum_dxhat = 0.0, sum_dxhat_x = 0.0;
            std::for_each(grad_output.begin(), grad_output.end(), [&, i = 0](double gout) mutable {
                double dxhat = gout * gamma[i];
                sum_dxhat += dxhat;
                sum_dxhat_x += dxhat * cache_x_hat[i];
                i++;
            });

            for (int i = 0; i < N; ++i) {
                double dxhat = grad_output[i] * gamma[i];
                grad_input[i] = (std_dev_inv / N) * (N * dxhat - sum_dxhat - cache_x_hat[i] * sum_dxhat_x);
            }

            return grad_input;
        }

        std::string name() const override { return "LayerNorm"; }
        bool has_params() const override { return true; }
        std::vector<double>& get_gamma() override { return gamma; }
        std::vector<double>& get_beta() override { return beta; }
        std::vector<double>& get_dgamma() override { return dgamma; }
        std::vector<double>& get_dbeta() override { return dbeta; }
    };

} // namespace nn

#endif