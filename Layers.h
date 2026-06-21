#ifndef LAYER_BACKPROP_H
#define LAYER_BACKPROP_H

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <utility>
#include <stdexcept>
#include <memory>
#include <iostream>
#include "activation_func.h"
#include "loss_func.h"
#include <vector>
namespace nn
{
    class BaseLayer
    {
    public:
        virtual ~BaseLayer() = default;

        // Core passes
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input) = 0;
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) = 0;

        // Utilities
        virtual void set_training_mode(bool mode) = 0;
        virtual int get_input_dim() const = 0;
        virtual int get_output_dim() const = 0;
        virtual std::string get_name() const = 0;

        // Weight accessors (Crucial for the Optimizer)
        // We add a check because some layers (like Dropout) don't have weights.
        virtual bool has_parameters() const { return false; }

        // These should throw an error or return dummy data if called on a parameter-free layer
        virtual Eigen::MatrixXd &get_weights() = 0;
        virtual Eigen::VectorXd &get_bias() = 0;
        virtual Eigen::MatrixXd &get_dw() = 0;
        virtual Eigen::VectorXd &get_db() = 0;
        virtual void zero_grad() { /* Do nothing by default */ }
        virtual ActivationFunction get_activation() const { return activations::Identity(); }
        virtual double get_dropout_rate() const { return 0.0; }
        virtual bool is_trainable() const { return true; }
        virtual void set_trainable(bool trainable) { /* Do nothing by default */ }
        virtual std::vector<int> get_hyperparams() const { return {}; }
        // --- Non-Trainable State (For BatchNorm) ---
        virtual std::vector<Eigen::VectorXd> get_running_stats() const { return {}; }
        virtual void set_running_stats(const std::vector<Eigen::VectorXd> &stats) { /* Do nothing by default */ }

        // for multithreading
        virtual std::unique_ptr<BaseLayer> clone() const = 0;
    };

    class DenseLayer : public BaseLayer
    {
    private:
        // --- Master Parameters (Shared across all clones) ---
        std::shared_ptr<Eigen::MatrixXd> weights;
        std::shared_ptr<Eigen::VectorXd> bias;

        // --- Thread-Local State (Unique to each clone) ---
        Eigen::MatrixXd dw;
        Eigen::VectorXd db;
        Eigen::MatrixXd input;
        Eigen::MatrixXd Z;

        int input_dim;
        int output_dim;
        ActivationFunction acti;
        bool trainable;
        bool use_bias; // The bypass flag

        // Private Constructor exclusively for clone() to inject shared pointers
        DenseLayer(int in_dim, int out_dim, ActivationFunction a,
                   std::shared_ptr<Eigen::MatrixXd> w, std::shared_ptr<Eigen::VectorXd> b, bool has_bias)
            : input_dim(in_dim), output_dim(out_dim), acti(std::move(a)),
              weights(w), bias(b), trainable(true), use_bias(has_bias)
        {
            dw = Eigen::MatrixXd::Zero(output_dim, input_dim);
            if (use_bias)
                db = Eigen::VectorXd::Zero(output_dim);
        }

    public:
        // Public Constructor for defining the model
        DenseLayer(int input_dim, int output_dim, ActivationFunction a, bool use_bias = true)
            : input_dim(input_dim), output_dim(output_dim), acti(std::move(a)),
              trainable(true), use_bias(use_bias)
        {
            // Allocate heap memory ONCE for the master layer
            weights = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Random(output_dim, input_dim) * 0.1);

            dw = Eigen::MatrixXd::Zero(output_dim, input_dim);

            if (use_bias)
            {
                bias = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(output_dim));
                db = Eigen::VectorXd::Zero(output_dim);
            }
            else
            {
                // Allocate dummy size 0 to prevent null pointer exceptions if accessed
                bias = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(0));
                db = Eigen::VectorXd::Zero(0);
            }
        }

        // --- Core Passes ---
        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            this->input = in;

            // Dereference the shared pointer to use the master matrix
            Z = input * (*weights).transpose();

            if (use_bias)
            {
                Z.rowwise() += (*bias).transpose();
            }

            return acti.forward(Z);
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            Eigen::MatrixXd act_grad = acti.backward(Z);
            Eigen::MatrixXd dZ = grad_output.cwiseProduct(act_grad);

            // Gradient for previous layer using shared master weights
            Eigen::MatrixXd grad_input = dZ * (*weights);

            if (trainable)
            {
                dw += dZ.transpose() * input;

                if (use_bias)
                {
                    db += dZ.colwise().sum().transpose();
                }
            }

            return grad_input;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override { /* Do nothing */ }
        int get_input_dim() const override { return input_dim; }
        int get_output_dim() const override { return output_dim; }
        std::string get_name() const override { return "Dense"; }

        // --- Weight Accessors ---
        // Return a reference to the dereferenced pointer to satisfy BaseLayer API!
        Eigen::MatrixXd &get_weights() override { return *weights; }
        Eigen::VectorXd &get_bias() override { return *bias; }
        Eigen::MatrixXd &get_dw() override { return dw; }
        Eigen::VectorXd &get_db() override { return db; }

        bool has_parameters() const override { return true; }
        ActivationFunction get_activation() const override { return acti; }
        bool is_trainable() const override { return trainable; }
        void set_trainable(bool t) override { trainable = t; }

        void zero_grad() override
        {
            dw.setZero();
            if (use_bias)
                db.setZero();
        }

        std::unique_ptr<BaseLayer> clone() const override
        {
            // Pass the shared_ptr directly. No deep copy is made!
            auto copy = std::unique_ptr<DenseLayer>(new DenseLayer(
                input_dim, output_dim, acti, weights, bias, use_bias));

            copy->set_trainable(this->trainable);
            return copy;
        }
    };
    class DropoutLayer : public BaseLayer
    {
    private:
        double rate;
        double scale; // Optimization: precompute the inverted dropout scale
        bool is_training = true;
        Eigen::MatrixXd binary_mask;

        std::mt19937 gen;

        // Dummy containers for the BaseLayer interface references
        Eigen::MatrixXd dummy_W;
        Eigen::VectorXd dummy_b;

    public:
        DropoutLayer(double dropout_rate) : rate(dropout_rate)
        {
            // Ensure rate is safe
            if (rate < 0.0)
                rate = 0.0;
            if (rate >= 1.0)
                rate = 0.99;

            // 1. Calculate the scale ONCE at layer creation
            scale = 1.0 / (1.0 - rate);

            std::random_device rd;
            gen.seed(rd());
        }

        // --- IDENTIFICATION ---
        bool has_parameters() const override { return false; }
        std::string get_name() const override { return "Dropout"; }
        double get_dropout_rate() const override { return rate; }

        // --- DIMENSIONS & MODES ---
        void set_training_mode(bool mode) override { is_training = mode; }
        int get_input_dim() const override { return 0; }
        int get_output_dim() const override { return 0; }

        // --- FORWARD PASS ---
        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override
        {
            if (!is_training || rate == 0.0)
                return input;

            std::bernoulli_distribution d(1.0 - rate);
            binary_mask.resize(input.rows(), input.cols());

            // 2. Faster memory access: Use 1D pointer traversal instead of 2D indexing
            double *mask_ptr = binary_mask.data();
            int total_elements = binary_mask.size();

            for (int i = 0; i < total_elements; ++i)
            {
                // 3. Bake the scale directly into the mask
                mask_ptr[i] = d(gen) ? scale : 0.0;
            }

            // 4. Instantly apply the mask. NO division required here anymore!
            return input.cwiseProduct(binary_mask);
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) override
        {
            if (!is_training || rate == 0.0)
                return grad;

            // 5. Instantly pass the gradient back. NO division required here anymore!
            return grad.cwiseProduct(binary_mask);
        }

        // --- PARAMETER GETTERS ---
        Eigen::MatrixXd &get_weights() override { return dummy_W; }
        Eigen::VectorXd &get_bias() override { return dummy_b; }
        Eigen::MatrixXd &get_dw() override { return dummy_W; }
        Eigen::VectorXd &get_db() override { return dummy_b; }

        std::unique_ptr<BaseLayer> clone() const override
        {
            auto cloned_layer = std::make_unique<DropoutLayer>(*this);

            std::random_device rd;
            cloned_layer->gen.seed(rd());

            return cloned_layer;
        }
    };

    class BatchNormalizationLayer : public BaseLayer
    {
    private:
        // --- Master Parameters (Shared across all clones) ---
        std::shared_ptr<Eigen::MatrixXd> gamma; // Scale [1 x Features]
        std::shared_ptr<Eigen::VectorXd> beta;  // Shift [Features x 1]

        // --- Thread-Local Gradients ---
        Eigen::MatrixXd dw;
        Eigen::VectorXd db;

        // --- Thread-Local Running Statistics ---
        Eigen::RowVectorXd running_mean;
        Eigen::RowVectorXd running_var;

        // --- Caches for Backward Pass ---
        Eigen::MatrixXd X_centered;
        Eigen::MatrixXd X_hat;
        Eigen::RowVectorXd std_dev_inv;

        int num_features;
        double momentum;
        double epsilon;
        bool is_training_mode;
        bool trainable;

        // Private Constructor for clone()
        BatchNormalizationLayer(int num_features, double momentum, double epsilon,
                                std::shared_ptr<Eigen::MatrixXd> g, std::shared_ptr<Eigen::VectorXd> b,
                                const Eigen::RowVectorXd &rm, const Eigen::RowVectorXd &rv, bool is_training)
            : num_features(num_features), momentum(momentum), epsilon(epsilon),
              gamma(g), beta(b), running_mean(rm), running_var(rv),
              is_training_mode(is_training), trainable(true)
        {
            dw = Eigen::MatrixXd::Zero(1, num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

    public:
        // Public Constructor
        BatchNormalizationLayer(int num_features, double momentum = 0.9, double epsilon = 1e-5)
            : num_features(num_features), momentum(momentum), epsilon(epsilon),
              is_training_mode(true), trainable(true)
        {
            // Allocate heap memory ONCE for the master layer
            gamma = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Ones(1, num_features));
            beta = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(num_features));

            dw = Eigen::MatrixXd::Zero(1, num_features);
            db = Eigen::VectorXd::Zero(num_features);

            running_mean = Eigen::RowVectorXd::Zero(num_features);
            running_var = Eigen::RowVectorXd::Ones(num_features);
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            if (in.cols() != num_features)
            {
                std::cerr << "\n[FATAL ERROR] BatchNormalization Dimension Mismatch!" << std::endl;
                exit(1);
            }

            int m = in.rows();
            Eigen::MatrixXd output(in.rows(), in.cols());

            if (is_training_mode)
            {
                Eigen::RowVectorXd batch_mean = in.colwise().mean();
                X_centered = in.rowwise() - batch_mean;

                Eigen::RowVectorXd batch_var = (X_centered.array().square().colwise().sum() / static_cast<double>(m)).matrix();
                std_dev_inv = (batch_var.array() + epsilon).rsqrt();
                X_hat = X_centered.array().rowwise() * std_dev_inv.array();

                running_mean = (momentum * running_mean).eval() + ((1.0 - momentum) * batch_mean).eval();
                running_var = (momentum * running_var).eval() + ((1.0 - momentum) * batch_var).eval();
            }
            else
            {
                Eigen::MatrixXd X_centered_inf = in.rowwise() - running_mean;
                Eigen::RowVectorXd std_dev_inv_inf = (running_var.array() + epsilon).rsqrt();
                X_hat = X_centered_inf.array().rowwise() * std_dev_inv_inf.array();
            }

            // Dereference shared pointers to scale and shift
            output = (X_hat.array().rowwise() * (*gamma).row(0).array()).rowwise() + (*beta).transpose().array();

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int m = grad_output.rows();

            Eigen::MatrixXd current_dw = (grad_output.cwiseProduct(X_hat)).colwise().sum();
            Eigen::VectorXd current_db = grad_output.colwise().sum().transpose();

            if (trainable)
            {
                dw += current_dw;
                db += current_db;
            }

            Eigen::ArrayXXd m_dY = m * grad_output.array();
            Eigen::ArrayXXd d_beta_broadcast = current_db.transpose().array().replicate(m, 1);
            Eigen::ArrayXXd X_hat_d_gamma = X_hat.array().rowwise() * current_dw.row(0).array();

            Eigen::ArrayXXd inner_term = m_dY - d_beta_broadcast - X_hat_d_gamma;

            // Dereference gamma shared pointer
            Eigen::ArrayXXd coeff = ((*gamma).row(0).array() * std_dev_inv.array()) / static_cast<double>(m);

            return inner_term.rowwise() * coeff.row(0);
        }

        void set_training_mode(bool mode) override { is_training_mode = mode; }
        int get_input_dim() const override { return num_features; }
        int get_output_dim() const override { return num_features; }
        std::string get_name() const override { return "BatchNormalization"; }

        // Return references to the dereferenced memory
        Eigen::MatrixXd &get_weights() override { return *gamma; }
        Eigen::VectorXd &get_bias() override { return *beta; }
        Eigen::MatrixXd &get_dw() override { return dw; }
        Eigen::VectorXd &get_db() override { return db; }

        bool has_parameters() const override { return true; }
        bool is_trainable() const override { return trainable; }
        void set_trainable(bool t) override { trainable = t; }

        void zero_grad() override
        {
            dw.setZero();
            db.setZero();
        }

        std::vector<Eigen::VectorXd> get_running_stats() const override
        {
            return {running_mean.transpose(), running_var.transpose()};
        }

        void set_running_stats(const std::vector<Eigen::VectorXd> &stats) override
        {
            if (stats.size() == 2)
            {
                running_mean = stats[0].transpose();
                running_var = stats[1].transpose();
            }
        }

        std::unique_ptr<BaseLayer> clone() const override
        {
            // Pass the shared_ptrs directly; local stats are deep-copied
            auto copy = std::unique_ptr<BatchNormalizationLayer>(new BatchNormalizationLayer(
                num_features, momentum, epsilon, gamma, beta, running_mean, running_var, is_training_mode));

            copy->set_trainable(this->trainable);
            return copy;
        }
    };

    class LayerNormalizationLayer : public BaseLayer
    {
    private:
        // --- Master Parameters ---
        std::shared_ptr<Eigen::MatrixXd> gamma;
        std::shared_ptr<Eigen::VectorXd> beta;

        // --- Thread-Local State ---
        Eigen::MatrixXd dw;
        Eigen::VectorXd db;
        Eigen::MatrixXd X_centered;
        Eigen::MatrixXd X_hat;
        Eigen::VectorXd std_dev_inv;

        int num_features;
        double epsilon;
        bool trainable;

        // Private Constructor for clone()
        LayerNormalizationLayer(int num_features, double epsilon,
                                std::shared_ptr<Eigen::MatrixXd> g, std::shared_ptr<Eigen::VectorXd> b)
            : num_features(num_features), epsilon(epsilon), gamma(g), beta(b), trainable(true)
        {
            dw = Eigen::MatrixXd::Zero(1, num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

    public:
        // Public Constructor
        LayerNormalizationLayer(int num_features, double epsilon = 1e-5)
            : num_features(num_features), epsilon(epsilon), trainable(true)
        {
            gamma = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Ones(1, num_features));
            beta = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(num_features));

            dw = Eigen::MatrixXd::Zero(1, num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            if (in.cols() != num_features)
            {
                std::cerr << "\n[FATAL ERROR] LayerNormalization Dimension Mismatch!" << std::endl;
                exit(1);
            }

            int F = in.cols();
            Eigen::VectorXd instance_mean = in.rowwise().mean();
            X_centered = in.colwise() - instance_mean;

            Eigen::VectorXd instance_var = (X_centered.array().square().rowwise().sum() / static_cast<double>(F)).matrix();
            std_dev_inv = (instance_var.array() + epsilon).rsqrt();
            X_hat = X_centered.array().colwise() * std_dev_inv.array();

            // Dereference shared parameters
            return (X_hat.array().rowwise() * (*gamma).row(0).array()).rowwise() + (*beta).transpose().array();
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int F = grad_output.cols();

            Eigen::MatrixXd current_dw = (grad_output.cwiseProduct(X_hat)).colwise().sum();
            Eigen::VectorXd current_db = grad_output.colwise().sum().transpose();

            if (trainable)
            {
                dw += current_dw;
                db += current_db;
            }

            // Dereference gamma
            Eigen::ArrayXXd dY_gamma = grad_output.array().rowwise() * (*gamma).row(0).array();

            Eigen::ArrayXd sum_dY_gamma = dY_gamma.rowwise().sum();
            Eigen::ArrayXd sum_dY_gamma_Xhat = (dY_gamma * X_hat.array()).rowwise().sum();

            Eigen::ArrayXXd inner_term = static_cast<double>(F) * dY_gamma;
            inner_term.colwise() -= sum_dY_gamma;
            inner_term -= X_hat.array().colwise() * sum_dY_gamma_Xhat;

            Eigen::ArrayXd coeff = std_dev_inv.array() / static_cast<double>(F);

            return inner_term.colwise() * coeff;
        }

        void set_training_mode(bool mode) override {}
        int get_input_dim() const override { return num_features; }
        int get_output_dim() const override { return num_features; }
        std::string get_name() const override { return "LayerNormalization"; }

        Eigen::MatrixXd &get_weights() override { return *gamma; }
        Eigen::VectorXd &get_bias() override { return *beta; }
        Eigen::MatrixXd &get_dw() override { return dw; }
        Eigen::VectorXd &get_db() override { return db; }

        bool has_parameters() const override { return true; }
        bool is_trainable() const override { return trainable; }
        void set_trainable(bool t) override { trainable = t; }
        void zero_grad() override
        {
            dw.setZero();
            db.setZero();
        }

        std::unique_ptr<BaseLayer> clone() const override
        {
            auto copy = std::unique_ptr<LayerNormalizationLayer>(
                new LayerNormalizationLayer(num_features, epsilon, gamma, beta));
            copy->set_trainable(this->trainable);
            return copy;
        }
    };

    class RMSNormalizationLayer : public BaseLayer
    {
    private:
        // --- Master Parameters ---
        std::shared_ptr<Eigen::MatrixXd> gamma;
        std::shared_ptr<Eigen::VectorXd> beta;

        // --- Thread-Local State ---
        Eigen::MatrixXd dw;
        Eigen::VectorXd db;
        Eigen::MatrixXd X_cache;
        Eigen::MatrixXd X_hat;
        Eigen::VectorXd inv_rms;

        int num_features;
        double epsilon;
        bool trainable;

        // Private Constructor for clone()
        RMSNormalizationLayer(int num_features, double epsilon,
                              std::shared_ptr<Eigen::MatrixXd> g, std::shared_ptr<Eigen::VectorXd> b)
            : num_features(num_features), epsilon(epsilon), gamma(g), beta(b), trainable(true)
        {
            dw = Eigen::MatrixXd::Zero(1, num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

    public:
        // Public Constructor
        RMSNormalizationLayer(int num_features, double epsilon = 1e-8)
            : num_features(num_features), epsilon(epsilon), trainable(true)
        {
            gamma = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Ones(1, num_features));
            beta = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(num_features));

            dw = Eigen::MatrixXd::Zero(1, num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            if (in.cols() != num_features)
            {
                std::cerr << "\n[FATAL ERROR] RMSNorm Dimension Mismatch!" << std::endl;
                exit(1);
            }

            X_cache = in;

            Eigen::VectorXd mean_sq = in.array().square().rowwise().mean();
            inv_rms = (mean_sq.array() + epsilon).rsqrt();
            X_hat = in.array().colwise() * inv_rms.array();

            // Dereference master parameters
            return (X_hat.array().rowwise() * (*gamma).row(0).array()).rowwise() + (*beta).transpose().array();
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int F = grad_output.cols();

            Eigen::MatrixXd current_dw = (grad_output.cwiseProduct(X_hat)).colwise().sum();
            Eigen::VectorXd current_db = grad_output.colwise().sum().transpose();

            if (trainable)
            {
                dw += current_dw;
                db += current_db;
            }

            // Dereference gamma
            Eigen::ArrayXXd dY_gamma = grad_output.array().rowwise() * (*gamma).row(0).array();

            Eigen::ArrayXd S = (dY_gamma * X_cache.array()).rowwise().sum();
            Eigen::ArrayXd R_sq_F = inv_rms.array().square() / static_cast<double>(F);

            Eigen::ArrayXXd inner_term = dY_gamma - X_cache.array().colwise() * (S * R_sq_F);

            return inner_term.colwise() * inv_rms.array();
        }

        void set_training_mode(bool mode) override {}
        int get_input_dim() const override { return num_features; }
        int get_output_dim() const override { return num_features; }
        std::string get_name() const override { return "RMSNorm"; }

        Eigen::MatrixXd &get_weights() override { return *gamma; }
        Eigen::VectorXd &get_bias() override { return *beta; }
        Eigen::MatrixXd &get_dw() override { return dw; }
        Eigen::VectorXd &get_db() override { return db; }

        bool has_parameters() const override { return true; }
        bool is_trainable() const override { return trainable; }
        void set_trainable(bool t) override { trainable = t; }
        void zero_grad() override
        {
            dw.setZero();
            db.setZero();
        }

        std::unique_ptr<BaseLayer> clone() const override
        {
            auto copy = std::unique_ptr<RMSNormalizationLayer>(
                new RMSNormalizationLayer(num_features, epsilon, gamma, beta));
            copy->set_trainable(this->trainable);
            return copy;
        }
    };

    // since this all the layers uses vectorxd so we actually doesn't need flatten but for shake of completeness i had created this.
    class FlattenLayer : public BaseLayer
    {
    private:
        int input_dim;
        int output_dim;

        // Dummy containers to safely return references if getters are accidentally called
        Eigen::MatrixXd dummy_W;
        Eigen::VectorXd dummy_b;

    public:
        // Constructor 1: If you know the 3D shape coming from a Conv Layer
        FlattenLayer(int channels, int height, int width)
        {
            input_dim = channels * height * width;
            output_dim = input_dim; // Flattening doesn't change total element count
        }

        // Constructor 2: If you just want to pass a flat size directly
        FlattenLayer(int flat_size)
        {
            input_dim = flat_size;
            output_dim = flat_size;
        }
        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            return in;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            return grad_output;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override { /* Parameterless, do nothing */ }

        int get_input_dim() const override { return input_dim; }
        int get_output_dim() const override { return output_dim; }
        std::string get_name() const override { return "Flatten"; }

        // --- Weight Accessors ---
        // We strictly declare this layer has NO parameters
        bool has_parameters() const override { return false; }

        // If the optimizer ignores has_parameters() and calls these anyway,
        // throwing an error prevents a silent segmentation fault.
        Eigen::MatrixXd &get_weights() override { throw std::runtime_error("Flatten has no weights"); }
        Eigen::VectorXd &get_bias() override { throw std::runtime_error("Flatten has no biases"); }
        Eigen::MatrixXd &get_dw() override { throw std::runtime_error("Flatten has no weights"); }
        Eigen::VectorXd &get_db() override { throw std::runtime_error("Flatten has no biases"); }
        std::unique_ptr<BaseLayer> clone() const override
        {
            // We can just use the flat_size constructor since input_dim == output_dim
            return std::make_unique<FlattenLayer>(this->input_dim);
        }
    };

    class Conv2DLayer : public BaseLayer
    {
    private:
        // --- Master Parameters (Shared across all clones) ---
        std::shared_ptr<Eigen::MatrixXd> W;
        std::shared_ptr<Eigen::VectorXd> b;

        // --- Thread-Local State (Unique to each clone) ---
        Eigen::MatrixXd dW;
        Eigen::VectorXd db;
        std::vector<Eigen::MatrixXd> col_caches;

        int in_channels, out_channels, kernel_size;
        int stride, padding;
        int in_h, in_w, out_h, out_w;

        bool trainable;
        bool use_bias; // Optimization flag

        // --- Helpers ---
        Eigen::MatrixXd im2col(const Eigen::VectorXd &image)
        {
            int patch_size = in_channels * kernel_size * kernel_size;
            int num_patches = out_h * out_w;
            Eigen::MatrixXd col_matrix = Eigen::MatrixXd::Zero(patch_size, num_patches);

            for (int ic = 0; ic < in_channels; ++ic)
            {
                for (int kh = 0; kh < kernel_size; ++kh)
                {
                    for (int kw = 0; kw < kernel_size; ++kw)
                    {
                        int row_idx = (ic * kernel_size * kernel_size) + (kh * kernel_size) + kw;
                        for (int oh = 0; oh < out_h; ++oh)
                        {
                            for (int ow = 0; ow < out_w; ++ow)
                            {
                                int col_idx = oh * out_w + ow;
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                {
                                    int img_idx = (ic * in_h * in_w) + (ih * in_w) + iw;
                                    col_matrix(row_idx, col_idx) = image(img_idx);
                                }
                            }
                        }
                    }
                }
            }
            return col_matrix;
        }

        Eigen::VectorXd col2im(const Eigen::MatrixXd &col_matrix)
        {
            Eigen::VectorXd image = Eigen::VectorXd::Zero(in_channels * in_h * in_w);

            for (int ic = 0; ic < in_channels; ++ic)
            {
                for (int kh = 0; kh < kernel_size; ++kh)
                {
                    for (int kw = 0; kw < kernel_size; ++kw)
                    {
                        int row_idx = (ic * kernel_size * kernel_size) + (kh * kernel_size) + kw;
                        for (int oh = 0; oh < out_h; ++oh)
                        {
                            for (int ow = 0; out_w > ow; ++ow)
                            {
                                int col_idx = oh * out_w + ow;
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                {
                                    int img_idx = (ic * in_h * in_w) + (ih * in_w) + iw;
                                    image(img_idx) += col_matrix(row_idx, col_idx);
                                }
                            }
                        }
                    }
                }
            }
            return image;
        }

        // Private Constructor exclusively for clone() to inject shared pointers
        Conv2DLayer(int in_c, int out_c, int ks, int s, int p, int ih, int iw,
                    std::shared_ptr<Eigen::MatrixXd> shared_W,
                    std::shared_ptr<Eigen::VectorXd> shared_b,
                    bool bias_flag, bool is_trainable)
            : in_channels(in_c), out_channels(out_c), kernel_size(ks),
              stride(s), padding(p), in_h(ih), in_w(iw),
              W(shared_W), b(shared_b), use_bias(bias_flag), trainable(is_trainable)
        {
            out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
            out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

            int fan_in = in_channels * kernel_size * kernel_size;
            dW = Eigen::MatrixXd::Zero(out_channels, fan_in);
            if (use_bias)
                db = Eigen::VectorXd::Zero(out_channels);
        }

    public:
        // Public Constructor
        Conv2DLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding, int in_h, int in_w, bool use_bias = true)
            : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
              stride(stride), padding(padding), in_h(in_h), in_w(in_w),
              trainable(true), use_bias(use_bias)
        {
            out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
            out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

            int fan_in = in_channels * kernel_size * kernel_size;

            // THE FIX: Xavier/Glorot Uniform Initialization (Bounded properly)
            double limit = std::sqrt(6.0 / (fan_in + out_channels));
            W = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Random(out_channels, fan_in) * limit);
            dW = Eigen::MatrixXd::Zero(out_channels, fan_in);

            if (use_bias)
            {
                b = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(out_channels));
                db = Eigen::VectorXd::Zero(out_channels);
            }
            else
            {
                b = std::make_shared<Eigen::VectorXd>(Eigen::VectorXd::Zero(0));
                db = Eigen::VectorXd::Zero(0);
            }
        }

        // --- Core Passes ---
        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override
        {
            int batch_size = input.rows();
            Eigen::MatrixXd output = Eigen::MatrixXd::Zero(batch_size, out_channels * out_h * out_w);

            col_caches.clear();
            col_caches.reserve(batch_size);

            for (int n = 0; n < batch_size; ++n)
            {
                Eigen::MatrixXd X_col = im2col(input.row(n));
                col_caches.push_back(X_col);

                // Dereference shared weights
                Eigen::MatrixXd out_col = (*W) * X_col;

                if (use_bias)
                {
                    out_col.colwise() += *b;
                }

                // ==========================================
                // THE FIX: Force memory into contiguous row-major format
                // before mapping to prevent channel interleaving/scrambling
                // ==========================================
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_out = out_col;

                Eigen::RowVectorXd flat_out = Eigen::Map<Eigen::RowVectorXd>(row_major_out.data(), row_major_out.size());
                output.row(n) = flat_out;
            }

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int batch_size = grad_output.rows();
            Eigen::MatrixXd grad_input = Eigen::MatrixXd::Zero(batch_size, in_channels * in_h * in_w);

            for (int n = 0; n < batch_size; ++n)
            {
                // Explicitly mapping as RowMajor accurately unpacks the gradient shape
                Eigen::MatrixXd d_out = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    grad_output.row(n).data(), out_channels, out_h * out_w);

                if (trainable)
                {
                    if (use_bias)
                    {
                        db += d_out.rowwise().sum();
                    }
                    dW += d_out * col_caches[n].transpose();
                }

                // Dereference shared weights
                Eigen::MatrixXd dX_col = (*W).transpose() * d_out;
                grad_input.row(n) = col2im(dX_col);
            }

            return grad_input;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override {}
        int get_input_dim() const override { return in_channels * in_h * in_w; }
        int get_output_dim() const override { return out_channels * out_h * out_w; }
        std::string get_name() const override { return "Conv2D"; }

        // --- Parameter Accessors ---
        bool has_parameters() const override { return true; }
        bool is_trainable() const override { return trainable; }
        void set_trainable(bool t) override { trainable = t; }

        Eigen::MatrixXd &get_weights() override { return *W; }
        Eigen::VectorXd &get_bias() override { return *b; }
        Eigen::MatrixXd &get_dw() override { return dW; }
        Eigen::VectorXd &get_db() override { return db; }

        void zero_grad() override
        {
            dW.setZero();
            if (use_bias)
                db.setZero();
        }

        std::vector<int> get_hyperparams() const override
        {
            return {in_channels, out_channels, kernel_size, stride, padding, in_h, in_w};
        }

        std::unique_ptr<BaseLayer> clone() const override
        {
            // Pass the shared pointers directly. No deep copy of massive convolution kernels!
            return std::unique_ptr<Conv2DLayer>(new Conv2DLayer(
                in_channels, out_channels, kernel_size, stride, padding, in_h, in_w,
                W, b, use_bias, trainable));
        }
    };

    class ActivationLayer : public BaseLayer
    {
    private:
        ActivationFunction activation_fn;
        Eigen::MatrixXd input_cache;

        // Dummy containers for the BaseLayer interface references
        Eigen::MatrixXd dummy_W;
        Eigen::VectorXd dummy_b;

    public:
        // Constructor takes ONLY the activation function object
        ActivationLayer(ActivationFunction act) : activation_fn(act) {}

        // --- IDENTIFICATION ---
        bool has_parameters() const override { return false; }

        // Dynamically returns the name of whatever activation was passed in
        std::string get_name() const override { return "ActivationLayer"; }

        // We override this so the model saver knows exactly which activation is inside
        ActivationFunction get_activation() const override { return activation_fn; }

        // --- DIMENSIONS & MODES ---
        void set_training_mode(bool mode) override { /* Parameterless */ }
        int get_input_dim() const override { return 0; }
        int get_output_dim() const override { return 0; }

        // --- FORWARD PASS ---
        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override
        {
            // Cache the input because the derivative usually requires it
            input_cache = input;
            return activation_fn.forward(input);
        }

        // --- BACKWARD PASS ---
        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) override
        {
            Eigen::MatrixXd act_derivative = activation_fn.backward(input_cache);
            return act_derivative.cwiseProduct(grad);
        }

        // --- PARAMETER GETTERS (Return empty dummies safely) ---
        Eigen::MatrixXd &get_weights() override { return dummy_W; }
        Eigen::VectorXd &get_bias() override { return dummy_b; }
        Eigen::MatrixXd &get_dw() override { return dummy_W; }
        Eigen::VectorXd &get_db() override { return dummy_b; }
        std::unique_ptr<BaseLayer> clone() const override
        {
            return std::make_unique<ActivationLayer>(this->activation_fn);
        }
    };
}
#endif