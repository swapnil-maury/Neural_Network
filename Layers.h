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