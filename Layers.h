#ifndef LAYER_BACKPROP_H
#define LAYER_BACKPROP_H

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <utility>
#include <stdexcept>

#include "activation_func.h"
#include "loss_func.h"

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
        // We add a check because some layers (like Dropout or MaxPool) don't have weights.
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
        virtual void set_running_stats(const std::vector<Eigen::VectorXd>& stats) { /* Do nothing by default */ }
    };

    class DenseLayer : public BaseLayer
    {
    private:
        Eigen::MatrixXd weights;
        Eigen::VectorXd bias;
        Eigen::MatrixXd input;
        Eigen::MatrixXd output;
        Eigen::MatrixXd Z;

        int input_dim;
        int output_dim;

        Eigen::MatrixXd dw;
        Eigen::VectorXd db;
        ActivationFunction acti;

        bool trainable;

    public:
        DenseLayer(int input_dim, int output_dim, ActivationFunction a)
            : acti(std::move(a)), input_dim(input_dim), output_dim(output_dim), trainable(true)
        {
            // Eigen Instant Initialization
            weights = Eigen::MatrixXd::Random(output_dim, input_dim) * 0.1;
            dw = Eigen::MatrixXd::Zero(output_dim, input_dim);

            bias = Eigen::VectorXd::Zero(output_dim);
            db = Eigen::VectorXd::Zero(output_dim);

            input = Eigen::VectorXd::Zero(input_dim);
            output = Eigen::VectorXd::Zero(output_dim);
            Z = Eigen::VectorXd::Zero(output_dim);
        }

        // --- Core Passes ---
        // Assuming 'acti' is an object of ActivationFunction

        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            this->input = in;

            // Pure Linear Algebra: Z = X * W^T + b
            // Use .rowwise() to add the bias vector to every row (image) in the batch
            Z = (input * weights.transpose()).rowwise() + bias.transpose();

            // Apply vectorized activation to the entire batch matrix Z
            output = acti.forward(Z);

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            // 1. Calculate activation gradients (dZ) using Hadamard product (element-wise multiplication)
            Eigen::MatrixXd act_grad = acti.backward(Z);
            Eigen::MatrixXd dZ = grad_output.cwiseProduct(act_grad);

            // 2. Calculate gradient to pass back to the previous layer
            Eigen::MatrixXd grad_input = dZ * weights;

            // 3. Update the layer's own gradients ONLY if it is trainable
            // 3. Calculate layer gradients over the ENTIRE batch instantly
            if (trainable)
            {
                // dW = dZ^T * X
                // dZ.T is [out_dim, N], input is [N, in_dim] -> Result: [out_dim, in_dim]
                Eigen::MatrixXd current_dw = dZ.transpose() * input;

                // db = Sum of dZ down the columns (over the batch dimension N)
                // Result is a row vector [1, out_dim], so we transpose it back to [out_dim, 1]
                Eigen::VectorXd current_db = dZ.colwise().sum().transpose();

                // Accumulate (so it works seamlessly with your optimizer logic)
                dw += current_dw;
                db += current_db;
            }

            return grad_input;
        }
        // --- Utilities ---

        // Required to satisfy BaseLayer's pure virtual function,
        // even though a purely mathematical DenseLayer ignores the training mode flag.
        void set_training_mode(bool mode) override { /* Do nothing */ }

        int get_input_dim() const override { return input_dim; }
        int get_output_dim() const override { return output_dim; }
        std::string get_name() const override { return "Dense"; }

        // --- Weight Accessors ---
        Eigen::MatrixXd &get_weights() override { return weights; }
        Eigen::VectorXd &get_bias() override { return bias; }
        Eigen::MatrixXd &get_dw() override { return dw; }
        Eigen::VectorXd &get_db() override { return db; }

        // --- Optional Overrides (Using default implementation signatures) ---
        bool has_parameters() const override { return true; }
        ActivationFunction get_activation() const override { return acti; }

        bool is_trainable() const override { return trainable; }
        void set_trainable(bool t) override { trainable = t; }
        void zero_grad() override
        {
            dw.setZero();
            db.setZero();
        }
    };

    class DropoutLayer : public BaseLayer
    {
    private:
        double rate;
        bool is_training = true;
        Eigen::MatrixXd binary_mask;

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
        }

        // --- IDENTIFICATION ---
        bool has_parameters() const override { return false; }
        std::string get_name() const override { return "Dropout"; }
        double get_dropout_rate() const override { return rate; }

        // --- DIMENSIONS & MODES ---
        void set_training_mode(bool mode) override { is_training = mode; }
        int get_input_dim() const override { return 0; }  // Not applicable for Dropout
        int get_output_dim() const override { return 0; } // Not applicable for Dropout

        // --- FORWARD PASS ---
        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override
        {
            if (!is_training || rate == 0.0)
                return input;

            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::bernoulli_distribution d(1.0 - rate);

            // Generate a random mask for the entire batch matrix
            binary_mask.resize(input.rows(), input.cols());
            for (int i = 0; i < binary_mask.rows(); ++i)
            {
                for (int j = 0; j < binary_mask.cols(); ++j)
                {
                    binary_mask(i, j) = d(gen) ? 1.0 : 0.0;
                }
            }

            return (input.cwiseProduct(binary_mask)) / (1.0 - rate);
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) override
        {
            if (!is_training || rate == 0.0)
                return grad;
            return (grad.cwiseProduct(binary_mask)) / (1.0 - rate);
        }

        // --- PARAMETER GETTERS (Return empty dummies safely) ---
        Eigen::MatrixXd &get_weights() override { return dummy_W; }
        Eigen::VectorXd &get_bias() override { return dummy_b; }
        Eigen::MatrixXd &get_dw() override { return dummy_W; }
        Eigen::VectorXd &get_db() override { return dummy_b; }
    };

    // i have to implement one more thing as batchnormalizationlayer will cancel out the biases so if we just at compile time just said
    // the denselayer to just stop making track or remove the biases from this denselayer then you will save thousands of calculation
    // and this will works and one more thing batch normalization should not palced before or after the dropout layer as it will cause variance and everything random so they are just fighting each other

    class BatchNormalizationLayer : public BaseLayer
    {
    private:
        // --- Learnable Parameters (Mapped to weights/bias for Optimizer) ---
        Eigen::MatrixXd gamma; // Scale (Stored as 1 x Features Matrix to match get_weights)
        Eigen::VectorXd beta;  // Shift (Stored as Features x 1 Vector to match get_bias)

        Eigen::MatrixXd dw; // Gradient of gamma
        Eigen::VectorXd db; // Gradient of beta

        // --- Running Statistics (Not trained by optimizer, tracked by layer) ---
        Eigen::RowVectorXd running_mean;
        Eigen::RowVectorXd running_var;

        // --- Caches for Backward Pass ---
        Eigen::MatrixXd X_centered;
        Eigen::MatrixXd X_hat;
        Eigen::RowVectorXd std_dev_inv;

        // --- Hyperparameters & State ---
        int num_features;
        double momentum;
        double epsilon;
        bool is_training_mode;
        bool trainable;

    public:
        BatchNormalizationLayer(int num_features, double momentum = 0.9, double epsilon = 1e-5)
            : num_features(num_features), momentum(momentum), epsilon(epsilon),
              is_training_mode(true), trainable(true)
        {
            // 1. Initialize Learnable Parameters
            // Gamma starts at 1.0 (no scaling initially)
            gamma = Eigen::MatrixXd::Ones(1, num_features);
            dw = Eigen::MatrixXd::Zero(1, num_features);

            // Beta starts at 0.0 (no shifting initially)
            beta = Eigen::VectorXd::Zero(num_features);
            db = Eigen::VectorXd::Zero(num_features);

            // 2. Initialize Running Stats
            running_mean = Eigen::RowVectorXd::Zero(num_features);
            running_var = Eigen::RowVectorXd::Ones(num_features);
        }

        // --- Core Passes ---
        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            // Safety Check: Instantly catch dimension mismatch errors before a core dump
            if (in.cols() != num_features)
            {
                std::cerr << "\n[FATAL ERROR] BatchNormalization Dimension Mismatch!" << std::endl;
                std::cerr << "Layer expected " << num_features << " features, but received " << in.cols() << " features." << std::endl;
                exit(1);
            }

            int m = in.rows(); // Batch size
            Eigen::MatrixXd output(in.rows(), in.cols());

            if (is_training_mode)
            {
                Eigen::RowVectorXd batch_mean = in.colwise().mean();
                X_centered = in.rowwise() - batch_mean;

                // Force explicit evaluation to prevent Eigen lazy-evaluation bugs under -O3
                Eigen::RowVectorXd batch_var = (X_centered.array().square().colwise().sum() / static_cast<double>(m)).matrix();
                std_dev_inv = (batch_var.array() + epsilon).rsqrt();
                X_hat = X_centered.array().rowwise() * std_dev_inv.array();

                // Force evaluation on the moving average addition
                running_mean = (momentum * running_mean).eval() + ((1.0 - momentum) * batch_mean).eval();
                running_var = (momentum * running_var).eval() + ((1.0 - momentum) * batch_var).eval();
            }
            else
            {
                // INFERENCE MODE
                Eigen::MatrixXd X_centered_inf = in.rowwise() - running_mean;
                Eigen::RowVectorXd std_dev_inv_inf = (running_var.array() + epsilon).rsqrt();
                X_hat = X_centered_inf.array().rowwise() * std_dev_inv_inf.array();
            }

            // Scale and Shift
            output = (X_hat.array().rowwise() * gamma.row(0).array()).rowwise() + beta.transpose().array();

            return output;
        }
        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int m = grad_output.rows(); // Batch size

            // 1. Calculate gradients for learnable parameters (Gamma and Beta)
            // d_gamma = sum of (dY * X_hat) down the columns
            Eigen::MatrixXd current_dw = (grad_output.cwiseProduct(X_hat)).colwise().sum();

            // d_beta = sum of dY down the columns
            Eigen::VectorXd current_db = grad_output.colwise().sum().transpose();

            if (trainable)
            {
                // Accumulate gradients for the optimizer
                dw += current_dw;
                db += current_db;
            }

            // 2. Calculate the gradient to pass back to the previous layer (dX)
            // Formula: dX = (gamma / (m * std_dev)) * [m * dY - sum(dY) - X_hat * sum(dY * X_hat)]

            // Prepare the broadcastable arrays
            Eigen::ArrayXXd m_dY = m * grad_output.array();
            Eigen::ArrayXXd d_beta_broadcast = current_db.transpose().array().replicate(m, 1);
            Eigen::ArrayXXd X_hat_d_gamma = X_hat.array().rowwise() * current_dw.row(0).array();

            // Calculate the inner term: (m * dY - d_beta - X_hat * d_gamma)
            Eigen::ArrayXXd inner_term = m_dY - d_beta_broadcast - X_hat_d_gamma;

            // Calculate the outer coefficient: (gamma / (m * std_dev))
            Eigen::ArrayXXd coeff = (gamma.row(0).array() * std_dev_inv.array()) / static_cast<double>(m);

            // Combine for final input gradient
            Eigen::MatrixXd grad_input = inner_term.rowwise() * coeff.row(0);

            return grad_input;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override { is_training_mode = mode; }
        int get_input_dim() const override { return num_features; }
        int get_output_dim() const override { return num_features; }
        std::string get_name() const override { return "BatchNormalization"; }

        // --- Parameter Accessors for Optimizer ---
        Eigen::MatrixXd &get_weights() override { return gamma; }
        Eigen::VectorXd &get_bias() override { return beta; }
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
        std::vector<Eigen::VectorXd> get_running_stats() const override {
            // Convert our RowVectors into standard VectorXd for the framework
            return {running_mean.transpose(), running_var.transpose()};
        }

        void set_running_stats(const std::vector<Eigen::VectorXd>& stats) override {
            if (stats.size() == 2) {
                // Convert the loaded VectorXd back into RowVectors
                running_mean = stats[0].transpose();
                running_var  = stats[1].transpose();
            }
        }
    };

    class LayerNormalizationLayer : public BaseLayer
    {
    private:
        // --- Learnable Parameters ---
        Eigen::MatrixXd gamma; // Scale [1 x Features]
        Eigen::VectorXd beta;  // Shift [Features x 1]

        Eigen::MatrixXd dw;
        Eigen::VectorXd db;

        // --- Caches for Backward Pass ---
        Eigen::MatrixXd X_centered;
        Eigen::MatrixXd X_hat;
        Eigen::VectorXd std_dev_inv; // Notice this is now a Column Vector [Batch x 1]

        int num_features;
        double epsilon;
        bool trainable;

    public:
        LayerNormalizationLayer(int num_features, double epsilon = 1e-5)
            : num_features(num_features), epsilon(epsilon), trainable(true)
        {
            gamma = Eigen::MatrixXd::Ones(1, num_features);
            dw = Eigen::MatrixXd::Zero(1, num_features);

            beta = Eigen::VectorXd::Zero(num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            // Safety Check
            if (in.cols() != num_features)
            {
                std::cerr << "\n[FATAL ERROR] LayerNormalization Dimension Mismatch!" << std::endl;
                exit(1);
            }

            int F = in.cols(); // Number of features

            // 1. Calculate Mean across the FEATURES (row-wise) -> Result is [Batch x 1]
            Eigen::VectorXd instance_mean = in.rowwise().mean();

            // 2. Center the data
            // We broadcast the column vector to subtract from every feature in the row
            X_centered = in.colwise() - instance_mean;

            // 3. Calculate Variance across the FEATURES
            Eigen::VectorXd instance_var = (X_centered.array().square().rowwise().sum() / static_cast<double>(F)).matrix();

            // 4. Calculate inverse standard dev
            std_dev_inv = (instance_var.array() + epsilon).rsqrt();

            // 5. Normalize
            X_hat = X_centered.array().colwise() * std_dev_inv.array();

            // 6. Scale and Shift (Gamma and Beta are applied to the features)
            Eigen::MatrixXd output = (X_hat.array().rowwise() * gamma.row(0).array()).rowwise() + beta.transpose().array();

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int M = grad_output.rows();
            int F = grad_output.cols();

            // 1. Calculate Gamma and Beta Gradients
            // Even though LayerNorm normalizes across features, gamma and beta are still
            // shared across the whole batch, so we sum down the columns just like BatchNorm.
            Eigen::MatrixXd current_dw = (grad_output.cwiseProduct(X_hat)).colwise().sum();
            Eigen::VectorXd current_db = grad_output.colwise().sum().transpose();

            if (trainable)
            {
                dw += current_dw;
                db += current_db;
            }

            // 2. Calculate Input Gradient (dX)
            // Because each row (instance) is normalized independently, we calculate dX row-by-row.

            // dY * gamma (broadcast gamma across the batch)
            Eigen::ArrayXXd dY_gamma = grad_output.array().rowwise() * gamma.row(0).array();

            // Sums across the features [Batch x 1 arrays]
            Eigen::ArrayXd sum_dY_gamma = dY_gamma.rowwise().sum();
            Eigen::ArrayXd sum_dY_gamma_Xhat = (dY_gamma * X_hat.array()).rowwise().sum();

            // Formula: dX = (1 / (F * std_dev)) * [ F * (dY * gamma) - sum(dY * gamma) - X_hat * sum(dY * gamma * X_hat) ]
            Eigen::ArrayXXd inner_term = static_cast<double>(F) * dY_gamma;
            inner_term.colwise() -= sum_dY_gamma; // Subtract row sums
            inner_term -= X_hat.array().colwise() * sum_dY_gamma_Xhat;

            Eigen::ArrayXd coeff = std_dev_inv.array() / static_cast<double>(F);

            // Multiply each row by its corresponding instance coefficient
            Eigen::MatrixXd grad_input = inner_term.colwise() * coeff;

            return grad_input;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override { /* LayerNorm doesn't care about training vs inference! */ }
        int get_input_dim() const override { return num_features; }
        int get_output_dim() const override { return num_features; }
        std::string get_name() const override { return "LayerNormalization"; }

        Eigen::MatrixXd &get_weights() override { return gamma; }
        Eigen::VectorXd &get_bias() override { return beta; }
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
    };

    class RMSNormalizationLayer : public BaseLayer
    {
    private:
        // --- Learnable Parameters ---
        Eigen::MatrixXd gamma; // Scale [1 x Features]
        Eigen::VectorXd beta;  // Shift [Features x 1] (Included for framework compatibility)

        Eigen::MatrixXd dw;
        Eigen::VectorXd db;

        // --- Caches for Backward Pass ---
        Eigen::MatrixXd X_cache; // Store raw input
        Eigen::MatrixXd X_hat;   // Store normalized input
        Eigen::VectorXd inv_rms; // Inverse Root Mean Square [Batch x 1]

        int num_features;
        double epsilon;
        bool trainable;

    public:
        RMSNormalizationLayer(int num_features, double epsilon = 1e-8)
            : num_features(num_features), epsilon(epsilon), trainable(true)
        {
            gamma = Eigen::MatrixXd::Ones(1, num_features);
            dw = Eigen::MatrixXd::Zero(1, num_features);

            beta = Eigen::VectorXd::Zero(num_features);
            db = Eigen::VectorXd::Zero(num_features);
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &in) override
        {
            // Safety Check
            if (in.cols() != num_features)
            {
                std::cerr << "\n[FATAL ERROR] RMSNorm Dimension Mismatch!" << std::endl;
                exit(1);
            }

            X_cache = in; // Save raw input for backprop

            // 1. Calculate Mean of Squares across the FEATURES (row-wise) -> [Batch x 1]
            // Notice we completely skip subtracting the mean here! This is why RMSNorm is faster.
            Eigen::VectorXd mean_sq = in.array().square().rowwise().mean();

            // 2. Calculate Inverse Root Mean Square
            inv_rms = (mean_sq.array() + epsilon).rsqrt();

            // 3. Normalize the input
            X_hat = in.array().colwise() * inv_rms.array();

            // 4. Scale and Shift
            Eigen::MatrixXd output = (X_hat.array().rowwise() * gamma.row(0).array()).rowwise() + beta.transpose().array();

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int F = grad_output.cols();

            // 1. Calculate Gamma and Beta Gradients
            Eigen::MatrixXd current_dw = (grad_output.cwiseProduct(X_hat)).colwise().sum();
            Eigen::VectorXd current_db = grad_output.colwise().sum().transpose();

            if (trainable)
            {
                dw += current_dw;
                db += current_db;
            }

            // 2. Calculate Input Gradient (dX) using RMSNorm Calculus

            // dY * gamma -> [Batch x Features]
            Eigen::ArrayXXd dY_gamma = grad_output.array().rowwise() * gamma.row(0).array();

            // S = Sum across features of (dY_gamma * X) -> [Batch x 1]
            Eigen::ArrayXd S = (dY_gamma * X_cache.array()).rowwise().sum();

            // R_sq_F = (inv_rms^2) / F -> [Batch x 1]
            Eigen::ArrayXd R_sq_F = inv_rms.array().square() / static_cast<double>(F);

            // inner_term = dY_gamma - X * (S * R_sq_F)
            Eigen::ArrayXXd inner_term = dY_gamma - X_cache.array().colwise() * (S * R_sq_F);

            // dX = inner_term * inv_rms
            Eigen::MatrixXd grad_input = inner_term.colwise() * inv_rms.array();

            return grad_input;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override { /* RMSNorm behaves the same in training and inference */ }
        int get_input_dim() const override { return num_features; }
        int get_output_dim() const override { return num_features; }
        std::string get_name() const override { return "RMSNorm"; }

        Eigen::MatrixXd &get_weights() override { return gamma; }
        Eigen::VectorXd &get_bias() override { return beta; }
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

        // --- Core Passes ---
        // Since the BaseLayer pipeline already passes 1D Eigen::VectorXd arrays,
        // Flatten just instantly passes the data forward to the Dense layer.
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
    };

    class Conv2DLayer : public BaseLayer
    {
    private:
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        int padding;
        int in_h, in_w;
        int out_h, out_w;

        Eigen::MatrixXd W; // Weights: (out_channels, in_channels * kernel_size * kernel_size)
        Eigen::VectorXd b; // Biases: (out_channels)

        Eigen::MatrixXd dW; // Gradient of weights
        Eigen::VectorXd db; // Gradient of biases

        // THE MISSING MEMBER: Stores the im2col matrix for each image in the batch
        std::vector<Eigen::MatrixXd> col_caches;

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
                            for (int ow = 0; ow < out_w; ++ow)
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

    public:
        Conv2DLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding, int in_h, int in_w)
            : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
              stride(stride), padding(padding), in_h(in_h), in_w(in_w)
        {
            out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
            out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

            int fan_in = in_channels * kernel_size * kernel_size;
            W = Eigen::MatrixXd::Random(out_channels, fan_in) * std::sqrt(2.0 / fan_in);
            b = Eigen::VectorXd::Zero(out_channels);

            dW = Eigen::MatrixXd::Zero(out_channels, fan_in);
            db = Eigen::VectorXd::Zero(out_channels);
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

                Eigen::MatrixXd out_col = W * X_col;
                out_col.colwise() += b;

                Eigen::RowVectorXd flat_out = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>>(out_col.data(), out_col.size());
                output.row(n) = flat_out;
            }

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output) override
        {
            int batch_size = grad_output.rows();
            Eigen::MatrixXd grad_input = Eigen::MatrixXd::Zero(batch_size, in_channels * in_h * in_w);

            zero_grad();

            for (int n = 0; n < batch_size; ++n)
            {
                Eigen::MatrixXd d_out = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    grad_output.row(n).data(), out_channels, out_h * out_w);

                db += d_out.rowwise().sum();
                dW += d_out * col_caches[n].transpose();

                Eigen::MatrixXd dX_col = W.transpose() * d_out;
                grad_input.row(n) = col2im(dX_col);
            }

            return grad_input;
        }

        // --- Utilities ---
        void set_training_mode(bool mode) override {}
        int get_input_dim() const override { return in_channels * in_h * in_w; }
        int get_output_dim() const override { return out_channels * out_h * out_w; }
        std::string get_name() const override { return "Conv2D"; }

        // --- Weight Accessors ---
        bool has_parameters() const override { return true; }
        Eigen::MatrixXd &get_weights() override { return W; }
        Eigen::VectorXd &get_bias() override { return b; }
        Eigen::MatrixXd &get_dw() override { return dW; }
        Eigen::VectorXd &get_db() override { return db; }

        void zero_grad() override
        {
            dW.setZero();
            db.setZero();
        }
        std::vector<int> get_hyperparams() const override
        {
            // We pack all the required constructor arguments into a single list
            return {in_channels, out_channels, kernel_size, stride, padding, in_h, in_w};
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

            // NOTE: Change ".forward" to whatever method your ActivationFunction uses
            // to calculate the math (e.g., .apply(input) or .calculate(input))
            return activation_fn.forward(input);
        }

        // --- BACKWARD PASS ---
        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) override
        {
            // The backward pass of an activation is the incoming gradient multiplied
            // element-wise by the derivative of the activation function.
            // NOTE: Change ".backward" or ".derivative" to match your codebase.
            Eigen::MatrixXd act_derivative = activation_fn.backward(input_cache);

            return act_derivative.cwiseProduct(grad);
        }

        // --- PARAMETER GETTERS (Return empty dummies safely) ---
        Eigen::MatrixXd &get_weights() override { return dummy_W; }
        Eigen::VectorXd &get_bias() override { return dummy_b; }
        Eigen::MatrixXd &get_dw() override { return dummy_W; }
        Eigen::VectorXd &get_db() override { return dummy_b; }

    };
}
#endif