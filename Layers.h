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

}
#endif