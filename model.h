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
#include <Eigen/Dense>
#include <iomanip>
#include <thread>
#include "Layers.h"
#include "loss_func.h"
#include "optimizer.h"
#include <Eigen/Core>

namespace nn
{

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
        int num_threads;

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
            if (index < 0 || index >= layers.size())
            {
                std::cerr << "Error: Invalid layer index for deletion!" << std::endl;
                return;
            }
            layers.erase(layers.begin() + index);
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

        std::vector<std::vector<double>> predict(const std::vector<std::vector<double>> &X)
        {
            int active_threads = (num_threads <= 0) ? std::thread::hardware_concurrency() : num_threads;
            Eigen::setNbThreads(active_threads);
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
        void fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int custom_batch_size = -1);
        void fit_multi_threaded(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int custom_batch_size = -1, int thread = -1);
        std::vector<std::unique_ptr<BaseLayer>> &get_layers() { return layers; }
        void set_epochs(int e) { epochs = e; }
        int get_epochs() const { return epochs; }
        void set_batch_size(int b) { batch_size = b; }
        int get_batch_size() const { return batch_size; }
        int get_input_dim() const { return input_dim; }
        void summary() const;
        void set_threads(int threads) { num_threads = threads; }
        int get_threads() const { return num_threads; }
        void save_binary(const std::string &filename = "model.bin") const;
        void load_binary(const std::string &filename);
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
        // out << std::setprecision(16) << std::scientific;

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
        for (size_t i = 0; i < layers.size(); ++i)
        {
            auto stats = layers[i]->get_running_stats();
            if (stats.empty())
            {
                out << "    {}";
            }
            else
            {
                out << "    {";
                for (size_t s = 0; s < stats.size(); ++s)
                {
                    out << "{";
                    for (int j = 0; j < stats[s].size(); ++j)
                    {
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

                if (i < p.layer_running_stats.size() && !p.layer_running_stats[i].empty())
                {
                    std::vector<Eigen::VectorXd> loaded_stats;
                    for (const auto &stat_vec : p.layer_running_stats[i])
                    {
                        Eigen::VectorXd vec(stat_vec.size());
                        for (size_t j = 0; j < stat_vec.size(); ++j)
                            vec(j) = stat_vec[j];
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
    inline void SequentialNetwork::fit(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int custom_batch_size)
    {
        int active_threads = (num_threads <= 0) ? std::thread::hardware_concurrency() : num_threads;
        Eigen::setNbThreads(active_threads);
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
    inline void SequentialNetwork::fit_multi_threaded(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int custom_batch_size, int thread_count)
    {
        // 1. Prevent Eigen internal oversubscription
        Eigen::setNbThreads(1);

        int active_threads = (thread_count <= 0) ? std::thread::hardware_concurrency() : thread_count;
        active_threads = std::min(active_threads, static_cast<int>(X.size()));
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

        // --- PRE-ALLOCATION ---
        // Clones share weights internally; only gradients/caches are separate
        std::vector<std::vector<std::unique_ptr<BaseLayer>>> worker_layers(active_threads);
        for (int t = 0; t < active_threads; ++t)
        {
            for (const auto &l : this->layers)
            {
                auto clone = l->clone();
                clone->set_training_mode(true);
                worker_layers[t].push_back(std::move(clone));
            }
        }

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double total_loss = 0.0;

            for (size_t b_start = 0; b_start < X.size(); b_start += eff_batch)
            {
                size_t b_end = std::min(X.size(), b_start + eff_batch);
                size_t c_batch_size = b_end - b_start;
                int num_threads = std::min(active_threads, static_cast<int>(c_batch_size));

                // 2. Zero out gradients natively (NO WEIGHT COPYING!)
                for (auto &l : layers)
                    l->zero_grad();
                for (int t = 0; t < num_threads; ++t)
                {
                    for (auto &l : worker_layers[t])
                        l->zero_grad();
                }

                std::vector<double> worker_losses(num_threads, 0.0);
                std::vector<std::thread> workers;

                int base_chunk = c_batch_size / num_threads;
                int remainder = c_batch_size % num_threads;
                int current_idx = b_start;

                // 3. Spawn Threads & Execute Parallel Pass
                for (int t = 0; t < num_threads; ++t)
                {
                    int local_batch = base_chunk + (t < remainder ? 1 : 0);
                    int start_idx = current_idx;
                    current_idx += local_batch;

                    workers.emplace_back([this, t, start_idx, local_batch, &X, &Y, &worker_layers, &worker_losses]()
                                         {
                        // A. Build Local Matrices mapping directly to memory
                        Eigen::MatrixXd X_local(local_batch, X[0].size());
                        Eigen::MatrixXd Y_local(local_batch, Y[0].size());
                        for (int i = 0; i < local_batch; ++i) {
                            X_local.row(i) = Eigen::Map<const Eigen::VectorXd>(X[start_idx + i].data(), X[0].size());
                            Y_local.row(i) = Eigen::Map<const Eigen::VectorXd>(Y[start_idx + i].data(), Y[0].size());
                        }

                        // B. Forward Pass (using shared weights)
                        Eigen::MatrixXd out = X_local;
                        for (auto& l : worker_layers[t]) {
                            out = l->forward(out);
                        }

                        // C. Compute Loss
                        LossFunction local_loss_fn = this->get_loss(this->loss.get_name());
                        worker_losses[t] = local_loss_fn.compute_loss(out, Y_local) * local_batch;

                        // D. Backward Pass (writes strictly to thread-local dw/db)
                        Eigen::MatrixXd grad = local_loss_fn.compute_gradient(out, Y_local);
                        for (int j = static_cast<int>(worker_layers[t].size()) - 1; j >= 0; --j) {
                            grad = worker_layers[t][j]->backward(grad);
                        } });
                }

                // 4. Join Threads
                for (auto &w : workers)
                {
                    if (w.joinable())
                        w.join();
                }

                // 5. Aggregate Gradients
                for (int t = 0; t < num_threads; ++t)
                {
                    total_loss += worker_losses[t];

                    for (size_t j = 0; j < layers.size(); ++j)
                    {
                        if (layers[j]->has_parameters() && layers[j]->is_trainable())
                        {
                            layers[j]->get_dw() += worker_layers[t][j]->get_dw();
                            layers[j]->get_db() += worker_layers[t][j]->get_db();
                        }
                    }
                }

                // 6. Safely Sync Running Statistics (Crucial for BatchNorm)
                for (size_t j = 0; j < layers.size(); ++j)
                {
                    auto stats = layers[j]->get_running_stats();
                    if (!stats.empty())
                    {
                        std::vector<Eigen::VectorXd> avg_stats = worker_layers[0][j]->get_running_stats();

                        for (int t = 1; t < num_threads; ++t)
                        {
                            auto thread_stats = worker_layers[t][j]->get_running_stats();
                            for (size_t s = 0; s < avg_stats.size(); ++s)
                            {
                                avg_stats[s] += thread_stats[s];
                            }
                        }

                        for (size_t s = 0; s < avg_stats.size(); ++s)
                        {
                            avg_stats[s] /= num_threads; // Average the variance and mean
                        }

                        layers[j]->set_running_stats(avg_stats);
                    }
                }

                // 7. Scale gradients by total batch size
                double inv_batch = 1.0 / c_batch_size;
                for (auto &l : layers)
                {
                    if (l->has_parameters() && l->is_trainable())
                    {
                        l->get_dw() *= inv_batch;
                        l->get_db() *= inv_batch;
                    }
                }

                // 8. Update Weights
                optimizer.step(layers);
            }

            if (epoch % (epochs / 10 + 1) == 0 || epoch == epochs - 1)
            {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << " | Multi-Thread Loss: " << total_loss / X.size() << std::endl;
            }
        }
    }
    inline void SequentialNetwork::save_binary(const std::string &filename) const
    {
        std::ofstream out(filename, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            std::cerr << "Error: Could not open " << filename << " for binary writing.\n";
            return;
        }

        // Binary I/O Lambda Helpers
        auto write_val = [&out](auto val)
        {
            out.write(reinterpret_cast<const char *>(&val), sizeof(val));
        };
        auto write_str = [&out, &write_val](const std::string &str)
        {
            size_t len = str.size();
            write_val(len);
            out.write(str.data(), len);
        };
        auto write_double_vec = [&out, &write_val](const std::vector<double> &vec)
        {
            size_t len = vec.size();
            write_val(len);
            if (len > 0)
                out.write(reinterpret_cast<const char *>(vec.data()), len * sizeof(double));
        };
        auto write_int_vec = [&out, &write_val](const std::vector<int> &vec)
        {
            size_t len = vec.size();
            write_val(len);
            if (len > 0)
                out.write(reinterpret_cast<const char *>(vec.data()), len * sizeof(int));
        };

        // 1. Core Metadata
        int mod_in_dim = (input_dim > 0) ? input_dim : (layers.empty() ? -1 : layers.front()->get_input_dim());
        write_val(mod_in_dim);
        write_val(epochs);
        write_val(batch_size);
        write_val(optimizer.get_timestep());
        write_val(optimizer.is_initialized());

        write_str(loss.get_name());
        write_str(optimizer.name);

        auto opt_params = optimizer.params();
        if (opt_params.empty())
            opt_params.push_back(optimizer.lr);
        write_double_vec(opt_params);

        // 2. Layers Data
        size_t num_layers = layers.size();
        write_val(num_layers);

        for (size_t i = 0; i < num_layers; ++i)
        {
            write_str(layers[i]->get_name());
            write_str(layers[i]->get_activation().get_name());
            write_val(layers[i]->get_dropout_rate());
            write_int_vec(layers[i]->get_hyperparams());

            bool has_p = layers[i]->has_parameters();
            write_val(has_p);

            if (has_p)
            {
                // Write Weights row-by-row
                const auto &W = layers[i]->get_weights();
                int r = W.rows(), c = W.cols();
                write_val(r);
                write_val(c);
                for (int rr = 0; rr < r; ++rr)
                {
                    for (int cc = 0; cc < c; ++cc)
                    {
                        double val = W(rr, cc);
                        write_val(val);
                    }
                }

                // Write Biases
                const auto &b = layers[i]->get_bias();
                int bs = b.size();
                write_val(bs);
                for (int j = 0; j < bs; ++j)
                {
                    double val = b(j);
                    write_val(val);
                }
            }

            // Write Running Stats (BatchNorm / RMSNorm)
            auto stats = layers[i]->get_running_stats();
            size_t n_stats = stats.size();
            write_val(n_stats);
            for (size_t s = 0; s < n_stats; ++s)
            {
                int sz = stats[s].size();
                write_val(sz);
                for (int j = 0; j < sz; ++j)
                {
                    double val = stats[s](j);
                    write_val(val);
                }
            }
        }

        // 3. Optimizer State W (4D Vector)
        const auto &state_W = optimizer.get_state_W();
        write_val(state_W.size());
        for (const auto &v1 : state_W)
        {
            write_val(v1.size());
            for (const auto &v2 : v1)
            {
                write_val(v2.size());
                for (const auto &v3 : v2)
                    write_double_vec(v3);
            }
        }

        // 4. Optimizer State b (3D Vector)
        const auto &state_b = optimizer.get_state_b();
        write_val(state_b.size());
        for (const auto &v1 : state_b)
        {
            write_val(v1.size());
            for (const auto &v2 : v1)
                write_double_vec(v2);
        }

        out.close();
        std::cout << "[Info] Model saved to binary format: " << filename
                  << " at timestep " << optimizer.get_timestep() << ".\n";
    }
    inline void SequentialNetwork::load_binary(const std::string &filename)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
        {
            std::cerr << "Error: Could not open " << filename << " for binary reading.\n";
            return;
        }

        // Binary I/O Lambda Helpers
        auto read_val = [&in]<typename T>(T &val)
        {
            in.read(reinterpret_cast<char *>(&val), sizeof(T));
        };
        auto read_size = [&in]() -> size_t
        {
            size_t s;
            in.read(reinterpret_cast<char *>(&s), sizeof(size_t));
            return s;
        };
        auto read_str = [&in, &read_size]() -> std::string
        {
            size_t len = read_size();
            std::string str(len, '\0');
            if (len > 0)
                in.read(&str[0], len);
            return str;
        };
        auto read_double_vec = [&in, &read_size]() -> std::vector<double>
        {
            size_t len = read_size();
            std::vector<double> vec(len);
            if (len > 0)
                in.read(reinterpret_cast<char *>(vec.data()), len * sizeof(double));
            return vec;
        };
        auto read_int_vec = [&in, &read_size]() -> std::vector<int>
        {
            size_t len = read_size();
            std::vector<int> vec(len);
            if (len > 0)
                in.read(reinterpret_cast<char *>(vec.data()), len * sizeof(int));
            return vec;
        };

        ModelParams p;

        // 1. Core Metadata
        read_val(p.input_dim);
        read_val(p.epochs);
        read_val(p.batch_size);
        read_val(p.optimizer_timestep);
        read_val(p.optimizer_initialized);

        p.loss = read_str();
        p.optimizer = read_str();
        p.optimizer_params = read_double_vec();

        // 2. Layers Data
        size_t num_layers = read_size();
        p.layer_names.resize(num_layers);
        p.activations.resize(num_layers);
        p.dropout_rates.resize(num_layers);
        p.layer_hparams.resize(num_layers);
        p.weights.resize(num_layers);
        p.biases.resize(num_layers);
        p.layer_running_stats.resize(num_layers);

        for (size_t i = 0; i < num_layers; ++i)
        {
            p.layer_names[i] = read_str();
            p.activations[i] = read_str();
            read_val(p.dropout_rates[i]);
            p.layer_hparams[i] = read_int_vec();

            bool has_p;
            read_val(has_p);

            if (has_p)
            {
                int r, c;
                read_val(r);
                read_val(c);
                p.weights[i].resize(r, std::vector<double>(c));
                for (int rr = 0; rr < r; ++rr)
                {
                    for (int cc = 0; cc < c; ++cc)
                    {
                        double val;
                        read_val(val);
                        p.weights[i][rr][cc] = val;
                    }
                }

                int bs;
                read_val(bs);
                p.biases[i].resize(bs);
                for (int j = 0; j < bs; ++j)
                {
                    double val;
                    read_val(val);
                    p.biases[i][j] = val;
                }
            }

            size_t n_stats = read_size();
            p.layer_running_stats[i].resize(n_stats);
            for (size_t s = 0; s < n_stats; ++s)
            {
                int sz;
                read_val(sz);
                p.layer_running_stats[i][s].resize(sz);
                for (int j = 0; j < sz; ++j)
                {
                    double val;
                    read_val(val);
                    p.layer_running_stats[i][s][j] = val;
                }
            }
        }

        // 3. Optimizer State W (4D Vector)
        size_t sw_s1 = read_size();
        p.optimizer_state_W.resize(sw_s1);
        for (size_t i = 0; i < sw_s1; ++i)
        {
            size_t sw_s2 = read_size();
            p.optimizer_state_W[i].resize(sw_s2);
            for (size_t j = 0; j < sw_s2; ++j)
            {
                size_t sw_s3 = read_size();
                p.optimizer_state_W[i][j].resize(sw_s3);
                for (size_t k = 0; k < sw_s3; ++k)
                {
                    p.optimizer_state_W[i][j][k] = read_double_vec();
                }
            }
        }

        // 4. Optimizer State b (3D Vector)
        size_t sb_s1 = read_size();
        p.optimizer_state_b.resize(sb_s1);
        for (size_t i = 0; i < sb_s1; ++i)
        {
            size_t sb_s2 = read_size();
            p.optimizer_state_b[i].resize(sb_s2);
            for (size_t j = 0; j < sb_s2; ++j)
            {
                p.optimizer_state_b[i][j] = read_double_vec();
            }
        }

        // Hand the populated struct off to the existing load function
        this->load_model(p);

        std::cout << "[Info] Binary model loaded successfully from " << filename << ".\n";
    }
} // namespace nn

#endif
