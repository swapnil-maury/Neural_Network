#ifndef MODEL_H
#define MODEL_H

#include <algorithm>
#include <cctype>
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
            if (key == "categoricalcrossentropy" || key == "cce") // NEW FIX
                return losses::SoftmaxCrossEntropy();
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
        std::ofstream out(filename);
        if (!out.is_open())
        {
            std::cerr << "Error: Could not open " << filename << " for writing.\n";
            return;
        }

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
                out << "    {}"; // Handle parameterless layers safely
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
                out << "    {}"; // Handle parameterless layers safely
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
    }
    inline void SequentialNetwork::load_model(const ModelParams &p)
    {
        layers.clear();
        input_dim = p.input_dim;
        epochs = p.epochs;
        batch_size = p.batch_size;
        loss = get_loss(p.loss);
        optimizer = get_optimizer(p.optimizer, p.optimizer_params);

        // FACTORY UPDATE: Loop through layer_names, not weights
        for (size_t i = 0; i < p.layer_names.size(); ++i)
        {
            std::string type = p.layer_names[i];

            // --- THE FIX ---
            // It is much safer to determine the layer's output dimension based on the BIAS vector.
            // For Dense: bias is size [output_dim].
            // For BN: beta (bias) is size [num_features].
            int out_d = (i < p.biases.size()) ? p.biases[i].size() : 0;

            // Input dimension still comes from the columns of the weight matrix (if it exists)
            int in_d = (i < p.weights.size() && !p.weights[i].empty()) ? p.weights[i][0].size() : 0;

            double d_rate = (i < p.dropout_rates.size()) ? p.dropout_rates[i] : 0.0;
            std::string act_name = (i < p.activations.size()) ? p.activations[i] : "identity";

            // ROUTE 1: Dense Layers
            if (type == "Dense" || type == "dense")
            {
                add_dense_layer(in_d, out_d, get_activation(act_name));

                // Only load weights if the layer actually expects them
                if (layers.back()->has_parameters() && out_d > 0 && in_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights();
                    auto &b_eigen = layers.back()->get_bias();
                    for (int r = 0; r < out_d; ++r)
                    {
                        b_eigen(r) = p.biases[i][r];
                        for (int c = 0; c < in_d; ++c)
                        {
                            W_eigen(r, c) = p.weights[i][r][c];
                        }
                    }
                }
            }
            // ROUTE 2: Dropout Layers (Parameterless)
            else if (type == "Dropout" || type == "dropout")
            {
                layers.push_back(std::make_unique<DropoutLayer>(d_rate));
                optimizer.set_initialized(false);
            }
            // ROUTE 3: Batch Normalization Layers
            else if (type == "BatchNormalization" || type == "batchnormalization")
            {
                // Note: out_d represents num_features in BN layer
                // Also, BN's weights (gamma) are stored as [1 x Features]
                layers.push_back(std::make_unique<BatchNormalizationLayer>(out_d));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights(); // This is Gamma
                    auto &b_eigen = layers.back()->get_bias();    // This is Beta

                    // b_eigen is size [Features]
                    for (int r = 0; r < out_d; ++r)
                    {
                        b_eigen(r) = p.biases[i][r];
                    }

                    // W_eigen (gamma) is size [1 x Features] in our implementation
                    // But standard JSON structure might store it as [1][Features]
                    if (!p.weights[i].empty())
                    {
                        for (int c = 0; c < out_d; ++c)
                        {
                            // Assuming it was saved as a 1-row matrix
                            W_eigen(0, c) = p.weights[i][0][c];
                        }
                    }
                }
            }
            // ROUTE 4: Layer Normalization Layers
            else if (type == "LayerNormalization" || type == "layernormalization")
            {
                // Note: out_d represents num_features in the LayerNorm layer
                layers.push_back(std::make_unique<LayerNormalizationLayer>(out_d));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights(); // This is Gamma [1 x Features]
                    auto &b_eigen = layers.back()->get_bias();    // This is Beta [Features]

                    // Load Beta (biases array is size [Features])
                    for (int r = 0; r < out_d; ++r)
                    {
                        b_eigen(r) = p.biases[i][r];
                    }

                    // Load Gamma (weights array is 2D, but we only use the first row [0][Features])
                    if (!p.weights[i].empty())
                    {
                        for (int c = 0; c < out_d; ++c)
                        {
                            W_eigen(0, c) = p.weights[i][0][c];
                        }
                    }
                }
            }

            else if (type == "RMSNorm" || type == "rmsnorm")
            {
                // out_d correctly pulls the number of features based on the biases logic we fixed earlier
                layers.push_back(std::make_unique<RMSNormalizationLayer>(out_d));
                optimizer.set_initialized(false);

                if (layers.back()->has_parameters() && out_d > 0)
                {
                    auto &W_eigen = layers.back()->get_weights(); // Gamma [1 x Features]
                    auto &b_eigen = layers.back()->get_bias();    // Beta [Features]

                    for (int r = 0; r < out_d; ++r)
                    {
                        b_eigen(r) = p.biases[i][r];
                    }

                    if (!p.weights[i].empty())
                    {
                        for (int c = 0; c < out_d; ++c)
                        {
                            W_eigen(0, c) = p.weights[i][0][c];
                        }
                    }
                }
            }
            // FALLBACK
            else
            {
                std::cerr << "Warning: Unknown layer type loaded: " << type << std::endl;
            }
        }

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