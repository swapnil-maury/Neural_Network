#ifndef MODEL_H
#define MODEL_H

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense> // NEW: Eigen included here
#include <iomanip>

#include "Layers.h"
#include "loss_func.h"
#include "optimizer.h"

namespace nn
{

    // Data Structure remains std::vector so static initialization (output.h) still works natively
    struct ModelParams
    {
        std::vector<std::string> layer_names;
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>> biases;
        std::vector<std::string> activations;
        std::vector<double> dropout_rates;
        std::vector<std::vector<int>> layer_hparams;
        std::vector<std::vector<std::vector<double>>> layer_running_stats;
        std::string loss;
        std::string optimizer;
        std::vector<double> optimizer_params;
        int input_dim;
        int epochs;
        int batch_size;
        int optimizer_timestep;
        std::vector<std::vector<std::vector<std::vector<double>>>> optimizer_state_W;
        std::vector<std::vector<std::vector<double>>> optimizer_state_b;
        bool optimizer_initialized;
    };

    class SequentialNetwork
    {
    private:
        std::vector<std::unique_ptr<BaseLayer>> layers;
        LossFunction loss;
        Optimizer optimizer;
        int epochs;
        int batch_size;
        int input_dim;

        static std::string canonical_key(const std::string &raw)
        {
            std::string key;
            for (char c : raw)
            {
                if (std::isalnum(static_cast<unsigned char>(c)))
                {
                    key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
            }
            return key;
        }

        ActivationFunction get_activation(const std::string &name, const std::vector<double> &p = {}) const
        {
            const std::string key = canonical_key(name);
            if (key == "identity")
                return activations::Identity();
            if (key == "binarystep")
                return activations::BinaryStep();
            if (key == "sigmoid")
                return activations::Sigmoid();
            if (key == "tanh")
                return activations::Tanh();
            if (key == "relu")
                return activations::ReLU();
            if (key == "relu6")
                return activations::ReLU6();
            if (key == "leakyrelu")
                return activations::LeakyReLU(p.empty() ? 0.01 : p[0]);
            if (key == "prelu")
                return activations::PReLU(p.empty() ? 0.25 : p[0]);
            if (key == "elu")
                return activations::ELU(p.empty() ? 1.0 : p[0]);
            if (key == "selu")
                return activations::SELU((p.size() > 0) ? p[0] : 1.0507009873554805, (p.size() > 1) ? p[1] : 1.6732632423543772);
            if (key == "softplus")
                return activations::Softplus();
            if (key == "softsign")
                return activations::Softsign();
            if (key == "swish")
                return activations::Swish(p.empty() ? 1.0 : p[0]);
            if (key == "silu")
                return activations::SiLU();
            if (key == "gelu")
                return activations::GELU();
            if (key == "mish")
                return activations::Mish();
            if (key == "hardsigmoid")
                return activations::HardSigmoid();
            if (key == "hardswish")
                return activations::HardSwish();
            if (key == "softmax") // NEW FIX
                return activations::Softmax();
            return activations::Identity();
        }

        LossFunction get_loss(const std::string &name, const std::vector<double> &p = {}) const
        {
            const std::string key = canonical_key(name);
            if (key == "mse")
                return losses::MSE();
            if (key == "mae")
                return losses::MAE();
            if (key == "binarycrossentropy" || key == "bce")
                return losses::BinaryCrossEntropy();
            if (key == "hinge")
                return losses::Hinge();
            if (key == "squaredhinge")
                return losses::SquaredHinge();
            if (key == "huber")
                return losses::Huber(p.empty() ? 1.0 : p[0]);
            if (key == "logcosh")
                return losses::LogCosh();
            if (key == "msle")
                return losses::MSLE();
            if (key == "poisson")
                return losses::Poisson();
            if (key == "kldivergence" || key == "kl")
                return losses::KLDivergence();
            if (key == "categoricalcrossentropy" || key == "cce")
                return losses::CategoricalCrossEntropy();
            if (key == "softmaxcrossentropy" || key == "softmaxce" || key == "sparsecategoricalcrossentropy")
                return losses::SoftmaxCrossEntropy();
            std::cerr << "[Warning] Unknown loss '" << name << "'. Falling back to MSE.\n";
            return losses::MSE();
        }

        Optimizer get_optimizer(const std::string &name, const std::vector<double> &p = {}) const
        {
            const std::string key = canonical_key(name);
            auto param = [&p](size_t idx, double f)
            { return (p.size() > idx) ? p[idx] : f; };
            if (key == "sgd")
                return nn::SGD(param(0, 0.01));
            if (key == "momentum")
                return nn::Momentum(param(0, 0.01), param(1, 0.9));
            if (key == "rmsprop")
                return nn::RMSProp(param(0, 0.001), param(1, 0.9), param(2, 1e-8));
            if (key == "adam")
                return nn::Adam(param(0, 0.001), param(1, 0.9), param(2, 0.999), param(3, 1e-8));
            return nn::SGD(0.01);
        }

        void set_training_mode(bool mode)
        {
            for (auto &l : layers)
            {
                l->set_training_mode(mode);
            }
        }

    public:
        SequentialNetwork(LossFunction loss_fn = losses::MSE(), Optimizer opt = nn::SGD(), int epochs = 100)
            : loss(std::move(loss_fn)), optimizer(std::move(opt)), epochs(epochs), batch_size(1), input_dim(-1) {}

        template <typename T>
        void add_layer(T layer_object)
        {
            // Moves the temporary object onto the heap and creates a unique_ptr
            layers.push_back(std::make_unique<T>(std::move(layer_object)));
            optimizer.set_initialized(false);
        }
        void add_dense_layer(int in_dim, int out_dim, ActivationFunction act)
        {
            layers.push_back(std::make_unique<DenseLayer>(in_dim, out_dim, std::move(act)));
            optimizer.set_initialized(false);
        }
        template <typename T>
        void insert_layer(int index, T layer_object)
        {
            if (index < 0 || index > layers.size())
            {
                std::cerr << "Error: Invalid layer index!" << std::endl;
                return;
            }

            // Moves the temporary object onto the heap, creates a unique_ptr, and inserts it
            layers.insert(layers.begin() + index, std::make_unique<T>(std::move(layer_object)));
            optimizer.set_initialized(false);
        }

        void delete_layer(int index)
        {
            // 1. Check if the index is valid
            // Note: Unlike insert, you cannot delete at layers.size(), so it must be >=
            if (index < 0 || index >= layers.size())
            {
                std::cerr << "Error: Invalid layer index for deletion!" << std::endl;
                return;
            }

            // 2. Erase the layer
            // Because you use std::unique_ptr, calling erase() automatically
            // calls the layer's destructor and frees the memory on the heap!
            layers.erase(layers.begin() + index);

            // 3. Reset the optimizer
            // The optimizer's momentum/velocity caches are now invalid, so it must reset.
            optimizer.set_initialized(false);

            std::cout << "Layer at index " << index << " successfully deleted." << std::endl;
        }
        // EIGEN UPDATE: Maps standard vectors to Eigen vectors instantly
        Eigen::MatrixXd forward(const Eigen::MatrixXd &input)
        {
            Eigen::MatrixXd out_eigen = input;
            for (auto &l : layers)
            {
                out_eigen = l->forward(out_eigen);
            }
            return out_eigen;
        }

        // Pass the entire batch gradient backwards instantly
        void backward(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y_true)
        {
            Eigen::MatrixXd current_grad = loss.compute_gradient(y_pred, y_true);

            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
            {
                current_grad = layers[i]->backward(current_grad);
            }
        }
        void fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int custom_batch_size = -1)
        {
            set_training_mode(true);
            int eff_batch = (custom_batch_size > 0) ? custom_batch_size : batch_size;

            if (!optimizer.is_initialized())
            {
                optimizer.set_timestep(0);
                std::cout << "[Info] Optimizer uninitialized (Architecture changed or fresh model). Resetting states and clock to 0.\n";
            }
            else
            {
                std::cout << "[Info] Resuming trained model perfectly from Timestep: " << optimizer.get_timestep() << "\n";
            }

            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                double total_loss = 0.0;

                for (size_t b_start = 0; b_start < X.size(); b_start += eff_batch)
                {
                    size_t b_end = std::min(X.size(), b_start + eff_batch);
                    size_t c_batch_size = b_end - b_start;

                    // 1. Build the 2D Batch Matrices
                    Eigen::MatrixXd X_batch(c_batch_size, X[0].size());
                    Eigen::MatrixXd Y_batch(c_batch_size, Y[0].size());

                    // Map the memory row-by-row into the batch matrix
                    for (size_t i = 0; i < c_batch_size; ++i)
                    {
                        X_batch.row(i) = Eigen::Map<const Eigen::VectorXd>(X[b_start + i].data(), X[0].size());
                        Y_batch.row(i) = Eigen::Map<const Eigen::VectorXd>(Y[b_start + i].data(), Y[0].size());
                    }

                    // 2. Clear previous batch gradients
                    for (auto &l : layers)
                    {
                        l->zero_grad();
                    }

                    // 3. ONE FORWARD PASS for the entire batch
                    Eigen::MatrixXd y_pred = forward(X_batch);

                    // Accumulate total loss for reporting
                    total_loss += loss.compute_loss(y_pred, Y_batch) * c_batch_size;

                    // 4. ONE BACKWARD PASS for the entire batch
                    backward(y_pred, Y_batch);

                    // 5. Average the accumulated layer gradients by batch size
                    double inv_batch = 1.0 / c_batch_size;
                    for (auto &l : layers)
                    {
                        if (l->has_parameters() && l->is_trainable())
                        {
                            l->get_dw() *= inv_batch;
                            l->get_db() *= inv_batch;
                        }
                    }

                    // 6. Update Weights
                    optimizer.step(layers);
                }

                if (epoch % (epochs / 10 + 1) == 0)
                {
                    std::cout << "Epoch " << epoch << " | Loss: " << total_loss / X.size() << std::endl;
                }
            }
        }
        std::vector<std::vector<double>> predict(const std::vector<std::vector<double>> &X)
        {
            set_training_mode(false);

            // Build the massive X matrix
            Eigen::MatrixXd X_batch(X.size(), X[0].size());
            for (size_t i = 0; i < X.size(); ++i)
            {
                X_batch.row(i) = Eigen::Map<const Eigen::VectorXd>(X[i].data(), X[0].size());
            }

            // INSTANT FORWARD PASS for all test data
            Eigen::MatrixXd out_eigen = forward(X_batch);

            // Convert back to std::vector for user output
            std::vector<std::vector<double>> preds(out_eigen.rows(), std::vector<double>(out_eigen.cols()));
            for (int i = 0; i < out_eigen.rows(); ++i)
            {
                Eigen::VectorXd row = out_eigen.row(i);
                preds[i] = std::vector<double>(row.data(), row.data() + row.size());
            }

            set_training_mode(true);
            return preds;
        }

        void load_model(const ModelParams &p);
        void save_model(const std::string &filename = "output.h") const;
        std::vector<std::unique_ptr<BaseLayer>> &get_layers() { return layers; }
        void set_epochs(int e) { epochs = e; }
        int get_epochs() const { return epochs; }
        void set_batch_size(int b) { batch_size = b; }
        int get_batch_size() const { return batch_size; }
        int get_input_dim() const { return input_dim; }
        void summary() const;
    };

    inline void SequentialNetwork::save_model(const std::string &filename) const
    {
        const std::string tmp_filename = filename + ".tmp";
        std::ofstream out(tmp_filename, std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            std::cerr << "Error: Could not open " << tmp_filename << " for writing: "
                      << std::strerror(errno) << "\n";
            return;
        }

        // CRITICAL FIX: Prevent float truncation to stop silent precision loss
        out << std::setprecision(16) << std::scientific;

        int mod_in_dim = (input_dim > 0) ? input_dim : (layers.empty() ? -1 : layers.front()->get_input_dim());

        out << "#ifndef MODEL_IMPORT\n#define MODEL_IMPORT\n\n"
            << "#include \"model.h\"\n\n"
            << "static nn::ModelParams model_param = {\n";

        // 0. Layer Names
        out << "  {";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            out << "\"" << layers[i]->get_name() << "\"" << (i + 1 < layers.size() ? ", " : "");
        }
        out << "},\n";

        // 1. Weights
        out << "  {\n";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i]->has_parameters())
            {
                const auto &W = layers[i]->get_weights();
                out << "    {";
                for (int r = 0; r < W.rows(); ++r)
                {
                    out << "{";
                    for (int c = 0; c < W.cols(); ++c)
                    {
                        out << W(r, c) << (c + 1 < W.cols() ? ", " : "");
                    }
                    out << "}" << (r + 1 < W.rows() ? ", " : "");
                }
                out << "}";
            }
            else
            {
                out << "    {}";
            }
            out << (i + 1 < layers.size() ? ",\n" : "\n");
        }
        out << "  },\n";

        // 2. Biases
        out << "  {\n";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i]->has_parameters())
            {
                const auto &b = layers[i]->get_bias();
                out << "    {";
                for (int j = 0; j < b.size(); ++j)
                {
                    out << b(j) << (j + 1 < b.size() ? ", " : "");
                }
                out << "}";
            }
            else
            {
                out << "    {}";
            }
            out << (i + 1 < layers.size() ? ",\n" : "\n");
        }
        out << "  },\n";

        // 3. Activations
        out << "  {";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            out << "\"" << layers[i]->get_activation().get_name() << "\"" << (i + 1 < layers.size() ? ", " : "");
        }
        out << "},\n";

        // 4. Dropout Rates
        out << "  {";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            out << layers[i]->get_dropout_rate() << (i + 1 < layers.size() ? ", " : "");
        }
        out << "},\n";

        // 4.5. Layer Hyperparameters (Essential for Conv2D and Flatten)
        out << "  {\n";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            auto hparams = layers[i]->get_hyperparams();
            if (hparams.empty())
            {
                out << "    {}";
            }
            else
            {
                out << "    {";
                for (size_t j = 0; j < hparams.size(); ++j)
                {
                    out << hparams[j] << (j + 1 < hparams.size() ? ", " : "");
                }
                out << "}";
            }
            out << (i + 1 < layers.size() ? ",\n" : "\n");
        }
        out << "  },\n";

        // ==========================================
        // 4.6 NEW SECTION: Layer Running Stats
        // ==========================================
        out << "  {\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            auto stats = layers[i]->get_running_stats();
            if (stats.empty()) {
                out << "    {}";
            } else {
                out << "    {";
                for (size_t s = 0; s < stats.size(); ++s) {
                    out << "{";
                    for (int j = 0; j < stats[s].size(); ++j) {
                        out << stats[s](j) << (j + 1 < stats[s].size() ? ", " : "");
                    }
                    out << "}" << (s + 1 < stats.size() ? ", " : "");
                }
                out << "}";
            }
            out << (i + 1 < layers.size() ? ",\n" : "\n");
        }
        out << "  },\n";

        // 5. Loss, Optimizer, and Params
        out << "  \"" << loss.get_name() << "\",\n  \"" << optimizer.name << "\",\n  {";
        auto opt_params = optimizer.params();
        if (opt_params.empty())
            opt_params.push_back(optimizer.lr);
        for (size_t i = 0; i < opt_params.size(); ++i)
        {
            out << opt_params[i] << (i + 1 < opt_params.size() ? ", " : "");
        }
        out << "},\n";

        // 6. Metadata
        out << "  " << mod_in_dim << ",\n  " << epochs << ",\n  " << batch_size << ",\n  " << optimizer.get_timestep() << ",\n";

        // 7. Optimizer State W
        out << "  {\n";
        const auto &state_W = optimizer.get_state_W();
        for (size_t i = 0; i < state_W.size(); ++i)
        {
            out << "    {\n";
            for (size_t r = 0; r < state_W[i].size(); ++r)
            {
                out << "      {";
                for (size_t c = 0; c < state_W[i][r].size(); ++c)
                {
                    out << "{";
                    for (size_t k = 0; k < state_W[i][r][c].size(); ++k)
                    {
                        out << state_W[i][r][c][k] << (k + 1 < state_W[i][r][c].size() ? ", " : "");
                    }
                    out << "}" << (c + 1 < state_W[i][r].size() ? ", " : "");
                }
                out << "}" << (r + 1 < state_W[i].size() ? ",\n" : "\n");
            }
            out << "    }" << (i + 1 < state_W.size() ? ",\n" : "\n");
        }
        out << "  },\n";

        // 8. Optimizer State B
        out << "  {\n";
        const auto &state_b = optimizer.get_state_b();
        for (size_t i = 0; i < state_b.size(); ++i)
        {
            out << "    {\n";
            for (size_t j = 0; j < state_b[i].size(); ++j)
            {
                out << "      {";
                for (size_t k = 0; k < state_b[i][j].size(); ++k)
                {
                    out << state_b[i][j][k] << (k + 1 < state_b[i][j].size() ? ", " : "");
                }
                out << "}" << (j + 1 < state_b[i].size() ? ",\n" : "\n");
            }
            out << "    }" << (i + 1 < state_b.size() ? ",\n" : "\n");
        }
        out << "  },\n";

        // 9. Initialized Boolean
        out << "  " << (optimizer.is_initialized() ? "true" : "false") << "\n};\n\n#endif\n";

        out.close();
        if (!out)
        {
            std::cerr << "Error: Failed while writing " << tmp_filename << ": "
                      << std::strerror(errno) << "\n";
            std::remove(tmp_filename.c_str());
            return;
        }

        if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0)
        {
            const int first_error = errno;
            if (std::remove(filename.c_str()) != 0 ||
                std::rename(tmp_filename.c_str(), filename.c_str()) != 0)
            {
                std::cerr << "Error: Could not replace " << filename << " with "
                          << tmp_filename << ": " << std::strerror(first_error) << "\n";
                return;
            }
        }

        std::cout << "[Info] Model saved to " << filename
                  << " at optimizer timestep " << optimizer.get_timestep() << ".\n";
    }
    inline void SequentialNetwork::load_model(const ModelParams &p)
    {
        layers.clear();
        input_dim = p.input_dim;
        epochs = p.epochs;
        batch_size = p.batch_size;
        loss = get_loss(p.loss);
        optimizer = get_optimizer(p.optimizer, p.optimizer_params);

        for (size_t i = 0; i < p.layer_names.size(); ++i)
        {
            std::string type = p.layer_names[i];

            // Safely grab parameter sizes
            int out_d = (i < p.biases.size()) ? p.biases[i].size() : 0;
            double d_rate = (i < p.dropout_rates.size()) ? p.dropout_rates[i] : 0.0;
            std::string act_name = (i < p.activations.size()) ? p.activations[i] : "identity";

            int previous_valid_dim = input_dim;
            for (int k = (int)layers.size() - 1; k >= 0; --k)
            {
                int d = layers[k]->get_output_dim();
                if (d > 0)
                {
                    previous_valid_dim = d;
                    break;
                }
            }

            // ROUTE 1: Dense Layers
            if (type == "Dense" || type == "dense")
            {
                add_dense_layer(previous_valid_dim, out_d, get_activation(act_name));

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights();
                    auto &b_eigen = layers.back()->get_bias();

                    // Copy exactly to Eigen boundaries
                    for (int r = 0; r < b_eigen.size(); ++r)
                    {
                        b_eigen(r) = p.biases[i][r];
                    }

                    if (!p.weights[i].empty())
                    {
                        for (int r = 0; r < W_eigen.rows(); ++r)
                        {
                            for (int c = 0; c < W_eigen.cols(); ++c)
                            {
                                W_eigen(r, c) = p.weights[i][r][c];
                            }
                        }
                    }
                }
            }

            // ROUTE 2: Dropout Layers
            else if (type == "Dropout" || type == "dropout")
            {
                layers.push_back(std::make_unique<DropoutLayer>(d_rate));
                optimizer.set_initialized(false);
            }

            // ROUTE 3: Batch Normalization
            else if (type == "BatchNormalization" || type == "batchnormalization")
            {
                layers.push_back(std::make_unique<BatchNormalizationLayer>(out_d));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights();
                    auto &b_eigen = layers.back()->get_bias();

                    for (int r = 0; r < out_d; ++r)
                        b_eigen(r) = p.biases[i][r];

                    if (!p.weights[i].empty())
                    {
                        for (int c = 0; c < out_d; ++c)
                            W_eigen(0, c) = p.weights[i][0][c];
                    }
                }
            
                if (i < p.layer_running_stats.size() && !p.layer_running_stats[i].empty()) {
                    std::vector<Eigen::VectorXd> loaded_stats;
                    for (const auto& stat_vec : p.layer_running_stats[i]) {
                        Eigen::VectorXd vec(stat_vec.size());
                        for (size_t j = 0; j < stat_vec.size(); ++j) vec(j) = stat_vec[j];
                        loaded_stats.push_back(vec);
                    }
                    layers.back()->set_running_stats(loaded_stats);
                }
            }

            // ROUTE 4: Layer Normalization
            else if (type == "LayerNormalization" || type == "layernormalization")
            {
                layers.push_back(std::make_unique<LayerNormalizationLayer>(out_d));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights();
                    auto &b_eigen = layers.back()->get_bias();

                    for (int r = 0; r < out_d; ++r)
                        b_eigen(r) = p.biases[i][r];

                    if (!p.weights[i].empty())
                    {
                        for (int c = 0; c < out_d; ++c)
                            W_eigen(0, c) = p.weights[i][0][c];
                    }
                }
            }

            // ROUTE 5: RMS Normalization
            else if (type == "RMSNorm" || type == "rmsnorm")
            {
                layers.push_back(std::make_unique<RMSNormalizationLayer>(out_d));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights();
                    auto &b_eigen = layers.back()->get_bias();

                    for (int r = 0; r < out_d; ++r)
                        b_eigen(r) = p.biases[i][r];

                    if (!p.weights[i].empty())
                    {
                        for (int c = 0; c < out_d; ++c)
                            W_eigen(0, c) = p.weights[i][0][c];
                    }
                }
            }

            // ROUTE 6: Conv2D Layers
            else if (type == "Conv2D" || type == "conv2d")
            {
                if (p.layer_hparams[i].size() != 7)
                {
                    throw std::runtime_error("Corrupted Model File: Conv2D requires 7 hyperparameters.");
                }

                int in_c = p.layer_hparams[i][0];
                int out_c = p.layer_hparams[i][1];
                int k_size = p.layer_hparams[i][2];
                int stride = p.layer_hparams[i][3];
                int pad = p.layer_hparams[i][4];
                int in_h = p.layer_hparams[i][5];
                int in_w = p.layer_hparams[i][6];

                layers.push_back(std::make_unique<Conv2DLayer>(in_c, out_c, k_size, stride, pad, in_h, in_w));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters())
                {
                    auto &W_eigen = layers.back()->get_weights();
                    auto &b_eigen = layers.back()->get_bias();
                    int fan_in = in_c * k_size * k_size;

                    for (int r = 0; r < out_c; ++r)
                    {
                        b_eigen(r) = p.biases[i][r];
                    }

                    if (!p.weights[i].empty())
                    {
                        for (int r = 0; r < out_c; ++r)
                        {
                            for (int c = 0; c < fan_in; ++c)
                            {
                                W_eigen(r, c) = p.weights[i][r][c];
                            }
                        }
                    }
                }
            }

            // ROUTE 7: Flatten Layers
            else if (type == "Flatten" || type == "flatten")
            {
                if (!p.layer_hparams[i].empty())
                {
                    layers.push_back(std::make_unique<FlattenLayer>(p.layer_hparams[i][0]));
                }
                else
                {
                    int previous_out_dim = layers.back()->get_output_dim();
                    layers.push_back(std::make_unique<FlattenLayer>(previous_out_dim));
                }
            }

            // ROUTE 8: Standalone Activation Layers
            else if (type == "ActivationLayer" || type == "activationlayer")
            {
                ActivationFunction act_fn = get_activation(act_name);
                layers.push_back(std::make_unique<ActivationLayer>(act_fn));
            }

            // FALLBACK
            else
            {
                std::cerr << "Warning: Unknown layer type loaded: " << type << std::endl;
            }
        }

        // Finalize Optimizer State
        optimizer.set_state_W(p.optimizer_state_W);
        optimizer.set_state_b(p.optimizer_state_b);
        optimizer.set_timestep(p.optimizer_timestep);
        optimizer.set_initialized(p.optimizer_initialized);
    }
    inline void SequentialNetwork::summary() const
    {
        std::cout << "Model: \"Sequential\"\n";
        std::cout << "_________________________________________________________________\n";
        std::cout << std::left << std::setw(29) << "Layer (type)"
                  << std::setw(26) << "Output Shape"
                  << "Param #\n";
        std::cout << "=================================================================\n";

        // Try to infer the starting dimension
        int current_dim = input_dim;
        if (current_dim <= 0 && !layers.empty() && layers.front()->has_parameters())
        {
            current_dim = layers.front()->get_input_dim();
        }

        std::string input_shape_str = "(None, " + std::to_string(current_dim) + ")";
        std::cout << std::left << std::setw(29) << "Input"
                  << std::setw(26) << input_shape_str
                  << "0\n";
        std::cout << "_________________________________________________________________\n";

        int total_params = 0;

        for (size_t i = 0; i < layers.size(); ++i)
        {
            const auto &l = layers[i];
            std::string type = l->get_name();
            std::string layer_name = type + "_" + std::to_string(i + 1);

            // Determine output shape (pass through if layer has no parameters)
            int out_dim = current_dim;
            if (l->has_parameters())
            {
                out_dim = l->get_output_dim();
            }

            // Calculate total weights + biases
            int params = 0;
            if (l->has_parameters())
            {
                const auto &W = l->get_weights();
                const auto &b = l->get_bias();
                params = (W.rows() * W.cols()) + b.size();
            }

            total_params += params;
            current_dim = out_dim; // carry over shape to the next layer

            // Print the beautifully formatted row
            std::string shape_str = "(None, " + std::to_string(out_dim) + ")";
            std::cout << std::left << std::setw(29) << layer_name
                      << std::setw(26) << shape_str
                      << params << "\n";

            if (i < layers.size() - 1)
            {
                std::cout << "_________________________________________________________________\n";
            }
        }

        std::cout << "=================================================================\n";
        // To add commas to large numbers (e.g., 1,000,000), you'd use a locale,
        // but standard printing works perfectly here.
        std::cout << "Total params: " << total_params << "\n";
        std::cout << "Trainable params: " << total_params << "\n";
        std::cout << "Non-trainable params: 0\n";
        std::cout << "_________________________________________________________________\n";
    }

} // namespace nn

#endif
