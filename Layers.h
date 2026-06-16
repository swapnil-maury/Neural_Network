#ifndef LAYER_BACKPROP_H
#define LAYER_BACKPROP_H

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <utility>

#include "activation_func.h"
#include "loss_func.h"

namespace nn
{
    class BaseLayer
    {
    public:
        virtual ~BaseLayer() = default;

        // Core passes
        virtual Eigen::VectorXd forward(const Eigen::VectorXd &input) = 0;
        virtual Eigen::VectorXd backward(const Eigen::VectorXd &grad) = 0;

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
        Eigen::VectorXd input;
        Eigen::VectorXd output;
        Eigen::VectorXd Z;

        int input_dim;
        int output_dim;

        Eigen::MatrixXd dw;
        Eigen::VectorXd db;
        ActivationFunction acti;

        double dropout_rate;
        Eigen::VectorXd mask;
        bool is_training;
        bool trainable;

        // Optimized Shared Random Engine
        static std::mt19937 &get_random_engine()
        {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            return gen;
        }

        double get_bernoulli(double keep_prob)
        {
            std::bernoulli_distribution dist(keep_prob);
            return dist(get_random_engine()) ? 1.0 : 0.0;
        }

    public:
        DenseLayer(int input_dim, int output_dim, ActivationFunction a, double dropout_rate = 0.0)
            : acti(std::move(a)), input_dim(input_dim), output_dim(output_dim),
              dropout_rate(dropout_rate), is_training(true), trainable(true)
        {

            // Eigen Instant Initialization
            // .Random() generates between -1 and 1. Multiplying by 0.1 scales it to [-0.1, 0.1]
            weights = Eigen::MatrixXd::Random(output_dim, input_dim) * 0.1;
            dw = Eigen::MatrixXd::Zero(output_dim, input_dim);

            bias = Eigen::VectorXd::Zero(output_dim);
            db = Eigen::VectorXd::Zero(output_dim);

            input = Eigen::VectorXd::Zero(input_dim);
            output = Eigen::VectorXd::Zero(output_dim);
            Z = Eigen::VectorXd::Zero(output_dim);
            mask = Eigen::VectorXd::Ones(output_dim);
        }

        void set_training_mode(bool mode) { is_training = mode; }
        double get_dropout_rate() const { return dropout_rate; }

        Eigen::VectorXd forward(const Eigen::VectorXd &in)
        {
            double keep_prob = 1.0 - dropout_rate;
            this->input = in;

            // Pure Linear Algebra: Z = W * X + b
            Z = (weights * input) + bias;

            for (int i = 0; i < output_dim; i++)
            {
                output[i] = acti.activate(Z[i]);

                if (dropout_rate > 0.0)
                {
                    if (is_training)
                    {
                        mask[i] = get_bernoulli(keep_prob);
                        output[i] = (output[i] * mask[i]) / keep_prob;
                    }
                    else
                    {
                        mask[i] = 1.0; // Pass-through during inference
                    }
                }
            }
            return output;
        }

        Eigen::VectorXd backward(const Eigen::VectorXd &grad_output) override
        {
            double keep_prob = 1.0 - dropout_rate;
            Eigen::VectorXd current_grad_output = grad_output;

            // 1. Apply dropout mask to incoming gradients
            if (dropout_rate > 0.0 && is_training)
            {
                for (int i = 0; i < output_dim; ++i)
                {
                    current_grad_output[i] = (current_grad_output[i] * mask[i]) / keep_prob;
                }
            }

            // 2. Calculate activation gradients into a local temporary vector (dZ)
            Eigen::VectorXd dZ(output_dim);
            for (int i = 0; i < output_dim; ++i)
            {
                dZ[i] = current_grad_output[i] * acti.grad(Z[i]);
            }

            // 3. Calculate gradient to pass back to the previous layer (ALWAYS happens)
            Eigen::VectorXd grad_input = weights.transpose() * dZ;

            // 4. Update the layer's own weights ONLY if it is not frozen
            if (trainable)
            {
                db = dZ;
                dw = dZ * input.transpose();
            }

            return grad_input;
        }

        // Updated Getters returning Eigen types by reference
        Eigen::MatrixXd &get_weights() { return this->weights; }
        Eigen::VectorXd &get_bias() { return this->bias; }
        Eigen::MatrixXd &get_dw() { return this->dw; }
        Eigen::VectorXd &get_db() { return this->db; }

        ActivationFunction get_activation() const { return this->acti; }
        int get_input_dim() const { return this->input_dim; }
        int get_output_dim() const { return this->output_dim; }

        void set_trainable(bool t) { trainable = t; }
        bool is_trainable() const { return trainable; }
        bool has_parameters() const override { return true; }
        std::string get_name() const override { return "Dense"; }
    };

    class DropoutLayer : public BaseLayer
    {
    private:
        double rate;
        bool is_training = true;
        Eigen::VectorXd binary_mask;

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
        Eigen::VectorXd forward(const Eigen::VectorXd &input) override
        {
            // If not training or rate is 0, pass data through untouched
            if (!is_training || rate == 0.0)
            {
                return input;
            }

            // Setup random number generator for Inverted Dropout
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::bernoulli_distribution d(1.0 - rate); // Probability of KEEPING a neuron

            binary_mask.resize(input.size());
            for (int i = 0; i < input.size(); ++i)
            {
                binary_mask(i) = d(gen) ? 1.0 : 0.0;
            }

            // Scale by 1/(1-rate) during training so we don't have to scale during prediction
            return (input.cwiseProduct(binary_mask)) / (1.0 - rate);
        }

        // --- BACKWARD PASS ---
        Eigen::VectorXd backward(const Eigen::VectorXd &grad) override
        {
            if (!is_training || rate == 0.0)
            {
                return grad;
            }
            // Gradients only flow back through the neurons that were kept active
            return (grad.cwiseProduct(binary_mask)) / (1.0 - rate);
        }

        // --- PARAMETER GETTERS (Return empty dummies safely) ---
        Eigen::MatrixXd &get_weights() override { return dummy_W; }
        Eigen::VectorXd &get_bias() override { return dummy_b; }
        Eigen::MatrixXd &get_dw() override { return dummy_W; }
        Eigen::VectorXd &get_db() override { return dummy_b; }
    };

}
#endif