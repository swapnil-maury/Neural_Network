#ifndef LAYER_BACKPROP_H
#define LAYER_BACKPROP_H

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include "activation_func.h"
#include "loss_func.h"

namespace nn {
    class layer {
    private:
        std::vector<std::vector<double>> weights;
        std::vector<double> bias;
        std::vector<double> input;
        std::vector<double> output;
        std::vector<double> Z;
        int input_dim;
        int output_dim;
        std::vector<std::vector<double>> dw;
        std::vector<double> db;
        ActivationFunction acti;
        
        // Modern random helper for weights
        double get_random_weight() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
        std::normal_distribution<double> dist(-0.1, 0.1);
            return dist(gen);
        }

    public:
        layer(int input_dim, int output_dim, ActivationFunction a) 
            : acti(std::move(a)), input_dim(input_dim), output_dim(output_dim) 
        {
            bias.assign(output_dim, 0.0);
            db.assign(output_dim,0.0);
            input.assign(input_dim, 0.0);
            output.assign(output_dim, 0.0);
            Z.assign(output_dim, 0.0);

            // Initialize with random weights
            weights.resize(output_dim, std::vector<double>(input_dim));
            dw.resize(output_dim, std::vector<double>(input_dim));

            for (auto& row : weights) {
                for (double& w : row) {
                    w = get_random_weight();
            }
        }
        }

        std::vector<double> forward(const std::vector<double>& in) {
            this->input = in;
            for (int i = 0; i < output_dim; i++) {
                double sum = 0;
                for (int j = 0; j < input_dim; j++) {
                    sum += weights[i][j] * in[j];
                }
                sum += bias[i];
                Z[i] = sum;
                output[i] = acti.activate(sum);
            }
            return output;
        }

        std::vector<double> backward(const std::vector<double>& grad_output) {
            for (int i = 0; i < output_dim; ++i) {
                    db[i] = grad_output[i] * acti.grad(Z[i]);
            }
            std::vector<double> grad_input(input_dim, 0.0);
            for (int j = 0; j < input_dim; ++j) {
                for (int i = 0; i < output_dim; ++i) {
                    grad_input[j] += weights[i][j] * db[i];
                }
            }
            for (int i = 0; i < output_dim; ++i) {
                for (int j = 0; j < input_dim; ++j) {
                    dw[i][j] = db[i]*input[j];
                }
            }
            return grad_input; 
        }

        // Added const and return by reference for efficiency
        std::vector<std::vector<double>>& get_weights(){
            return this->weights;
        }
        std::vector<double>& get_bias(){
            return this->bias;
        }
        std::vector<std::vector<double>> & get_dw(){
            return this->dw;
        }
        std::vector<double> & get_db(){
            return this->db;
        }
        ActivationFunction get_activation() const {
            return this->acti;
        }
        int get_input_dim()const{
            return this->input_dim;
        }
        int get_output_dim()const{
            return this->output_dim;
        }
    };
}
#endif