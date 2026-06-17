#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>

class ActivationFunction {
private:
    // Functions now take and return Eigen::MatrixXd
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation_;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative_;
    std::string name_;

public:
    ActivationFunction(std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation,
                       std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> derivative,
                       std::string name = "custom")
        : activation_(std::move(activation)),
          derivative_(std::move(derivative)),
          name_(std::move(name)) {}

    // Vectorized forward pass
    Eigen::MatrixXd forward(const Eigen::MatrixXd& z) const {
        return activation_(z);
    }

    // Vectorized backward pass
    Eigen::MatrixXd backward(const Eigen::MatrixXd& z) const {
        return derivative_(z);
    }

    const std::string& get_name() const {
        return name_;
    }
};

namespace activations {

inline ActivationFunction Identity() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) { return z; },
        [](const Eigen::MatrixXd& z) { 
            // FIX: Use rows() and cols() for 2D matrices
            return Eigen::MatrixXd::Ones(z.rows(), z.cols()); 
        },
        "identity"
    );
}

inline ActivationFunction BinaryStep() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : 0.0; });
        },
        [](const Eigen::MatrixXd& z) { 
            // FIX: Use rows() and cols() for 2D matrices
            return Eigen::MatrixXd::Zero(z.rows(), z.cols()); 
        },
        "binary_step"
    );
}

inline ActivationFunction Sigmoid() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                const double s = 1.0 / (1.0 + std::exp(-v));
                return s * (1.0 - s);
            });
        },
        "sigmoid"
    );
}

inline ActivationFunction Tanh() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.array().tanh().matrix(); // Native Eigen is extremely fast here
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                const double t = std::tanh(v);
                return 1.0 - t * t;
            });
        },
        "tanh"
    );
}

inline ActivationFunction ReLU() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.cwiseMax(0.0); // Native Eigen optimized
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return v > 0.0 ? 1.0 : 0.0; });
        },
        "relu"
    );
}

inline ActivationFunction ReLU6() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return std::min(6.0, std::max(0.0, v)); });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return (v > 0.0 && v < 6.0) ? 1.0 : 0.0; });
        },
        "relu6"
    );
}

inline ActivationFunction LeakyReLU(double alpha = 0.01) {
    return ActivationFunction(
        [alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([alpha](double v) { return v > 0.0 ? v : alpha * v; });
        },
        [alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([alpha](double v) { return v > 0.0 ? 1.0 : alpha; });
        },
        "leaky_relu"
    );
}

inline ActivationFunction PReLU(double alpha = 0.25) {
    return ActivationFunction(
        [alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([alpha](double v) { return v > 0.0 ? v : alpha * v; });
        },
        [alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([alpha](double v) { return v > 0.0 ? 1.0 : alpha; });
        },
        "prelu"
    );
}

inline ActivationFunction ELU(double alpha = 1.0) {
    return ActivationFunction(
        [alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([alpha](double v) { return v >= 0.0 ? v : alpha * (std::exp(v) - 1.0); });
        },
        [alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([alpha](double v) { return v >= 0.0 ? 1.0 : alpha * std::exp(v); });
        },
        "elu"
    );
}

inline ActivationFunction SELU(double lambda = 1.0507009873554805, double alpha = 1.6732632423543772) {
    return ActivationFunction(
        [lambda, alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([lambda, alpha](double v) {
                return lambda * (v >= 0.0 ? v : alpha * (std::exp(v) - 1.0));
            });
        },
        [lambda, alpha](const Eigen::MatrixXd& z) {
            return z.unaryExpr([lambda, alpha](double v) {
                return lambda * (v >= 0.0 ? 1.0 : alpha * std::exp(v));
            });
        },
        "selu"
    );
}

inline ActivationFunction Softplus() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                if (v > 20.0) return v;
                if (v < -20.0) return std::exp(v);
                return std::log1p(std::exp(v));
            });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
        },
        "softplus"
    );
}

inline ActivationFunction Softsign() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) { return v / (1.0 + std::abs(v)); });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                const double d = 1.0 + std::abs(v);
                return 1.0 / (d * d);
            });
        },
        "softsign"
    );
}

inline ActivationFunction Swish(double beta = 1.0) {
    return ActivationFunction(
        [beta](const Eigen::MatrixXd& z) {
            return z.unaryExpr([beta](double v) {
                const double s = 1.0 / (1.0 + std::exp(-beta * v));
                return v * s;
            });
        },
        [beta](const Eigen::MatrixXd& z) {
            return z.unaryExpr([beta](double v) {
                const double s = 1.0 / (1.0 + std::exp(-beta * v));
                return s + beta * v * s * (1.0 - s);
            });
        },
        "swish"
    );
}

inline ActivationFunction SiLU() {
    return Swish(1.0);
}

inline ActivationFunction GELU() {
    constexpr double kInvSqrt2 = 0.7071067811865475;
    constexpr double kInvSqrt2Pi = 0.3989422804014327;

    return ActivationFunction(
        [kInvSqrt2](const Eigen::MatrixXd& z) {
            return z.unaryExpr([kInvSqrt2](double v) {
                return 0.5 * v * (1.0 + std::erf(v * kInvSqrt2));
            });
        },
        [kInvSqrt2, kInvSqrt2Pi](const Eigen::MatrixXd& z) {
            return z.unaryExpr([kInvSqrt2, kInvSqrt2Pi](double v) {
                return 0.5 * (1.0 + std::erf(v * kInvSqrt2)) + v * kInvSqrt2Pi * std::exp(-0.5 * v * v);
            });
        },
        "gelu"
    );
}

inline ActivationFunction Mish() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                const double sp = v > 20.0 ? v : std::log1p(std::exp(v));
                return v * std::tanh(sp);
            });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                const double sp = v > 20.0 ? v : std::log1p(std::exp(v));
                const double t = std::tanh(sp);
                const double s = 1.0 / (1.0 + std::exp(-v));
                return t + v * s * (1.0 - t * t);
            });
        },
        "mish"
    );
}

inline ActivationFunction HardSigmoid() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                return std::min(1.0, std::max(0.0, (v + 3.0) / 6.0));
            });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                return (v > -3.0 && v < 3.0) ? (1.0 / 6.0) : 0.0;
            });
        },
        "hard_sigmoid"
    );
}

inline ActivationFunction HardSwish() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                const double hs = std::min(1.0, std::max(0.0, (v + 3.0) / 6.0));
                return v * hs;
            });
        },
        [](const Eigen::MatrixXd& z) {
            return z.unaryExpr([](double v) {
                if (v <= -3.0) return 0.0;
                if (v >= 3.0) return 1.0;
                return v / 3.0 + 0.5;
            });
        },
        "hard_swish"
    );
}

inline ActivationFunction Softmax() {
    return ActivationFunction(
        [](const Eigen::MatrixXd& z) -> Eigen::MatrixXd {
            if (z.size() == 0) return Eigen::MatrixXd();
            
            // 1. Get max logit PER ROW (per image in the batch for stability)
            // Result is a column vector [Batch_Size x 1]
            Eigen::VectorXd max_logits = z.rowwise().maxCoeff();
            
            // 2. Subtract max logit from each row, then exp()
            // .colwise() allows us to subtract the [Batch x 1] vector from all 10 columns
            Eigen::MatrixXd exp_vals = (z.colwise() - max_logits).array().exp().matrix();
            
            // 3. Get sum of exponentials PER ROW
            Eigen::VectorXd sums = exp_vals.rowwise().sum();
            
            // 4. Divide each element by its row's sum to get final probabilities
            return exp_vals.array().colwise() / sums.array();
        },
        [](const Eigen::MatrixXd& z) -> Eigen::MatrixXd {
            // Return a matrix of 1s matching the batch dimensions
            return Eigen::MatrixXd::Ones(z.rows(), z.cols());
        },
        "softmax"
    );
}

} // namespace activations

#endif