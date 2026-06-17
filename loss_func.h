#ifndef LOSS_FUNC_H
#define LOSS_FUNC_H

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <utility>
#include <cmath>
#include <stdexcept>

class LossFunction {
private:
    std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss_fn_;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> deriv_fn_;
    std::string name_;

public:
    LossFunction(std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss_fn,
                 std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> deriv_fn,
                 std::string name = "custom")
        : loss_fn_(std::move(loss_fn)),
          deriv_fn_(std::move(deriv_fn)),
          name_(std::move(name)) {}

    double compute_loss(const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) const {
        return loss_fn_(y_pred, y_true);
    }

    Eigen::MatrixXd compute_gradient(const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) const {
        return deriv_fn_(y_pred, y_true);
    }

    const std::string& get_name() const {
        return name_;
    }
};

namespace losses {

// Small epsilon for numerical stability
inline constexpr double kEps = 1e-12;

// Helper to keep probabilities in (0, 1) range natively in Eigen
// Update clamp_prob to handle and return 2D matrices
inline Eigen::MatrixXd clamp_prob(const Eigen::MatrixXd& p) {
    return p.array().cwiseMax(kEps).cwiseMin(1.0 - kEps).matrix();
}


// --- Scalar Loss Functions (Vectorized over elements) ---

inline LossFunction MSE() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return (y_pred - y_true).squaredNorm() / y_pred.size(); // Mean loss
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return 2.0 * (y_pred - y_true); // Raw gradient (no size division, matching your logic)
        },
        "MSE"
    );
}

inline LossFunction MAE() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return (y_pred - y_true).cwiseAbs().mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return (y_pred - y_true).array().sign().matrix();
        },
        "MAE"
    );
}

inline LossFunction BinaryCrossEntropy() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = clamp_prob(y_pred);
            Eigen::ArrayXd t = y_true.array();
            return -(t * p.log() + (1.0 - t) * (1.0 - p).log()).mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = clamp_prob(y_pred);
            return ((p - y_true.array()) / (p * (1.0 - p))).matrix();
        },
        "BinaryCrossEntropy"
    );
}

inline LossFunction Hinge() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return (1.0 - y_true.array() * y_pred.array()).cwiseMax(0.0).mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd margin = 1.0 - y_true.array() * y_pred.array();
            return ((margin > 0.0).cast<double>() * -y_true.array()).matrix();
        },
        "Hinge"
    );
}

inline LossFunction SquaredHinge() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd margin = (1.0 - y_true.array() * y_pred.array()).cwiseMax(0.0);
            return margin.square().mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd margin = 1.0 - y_true.array() * y_pred.array();
            return ((margin > 0.0).cast<double>() * -2.0 * y_true.array() * margin).matrix();
        },
        "SquaredHinge"
    );
}

inline LossFunction Huber(double delta = 1.0) {
    return LossFunction(
        [delta](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd e = y_pred.array() - y_true.array();
            Eigen::ArrayXd ae = e.abs();
            return ((ae <= delta).cast<double>() * (0.5 * e.square()) + 
                    (ae > delta).cast<double>() * (delta * (ae - 0.5 * delta))).mean();
        },
        [delta](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd e = y_pred.array() - y_true.array();
            Eigen::ArrayXd ae = e.abs();
            return ((ae <= delta).cast<double>() * e + 
                    (ae > delta).cast<double>() * e.sign() * delta).matrix();
        },
        "Huber"
    );
}

inline LossFunction LogCosh() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return (y_pred - y_true).array().cosh().log().mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            return (y_pred - y_true).array().tanh().matrix();
        },
        "LogCosh"
    );
}

inline LossFunction MSLE() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = y_pred.array().cwiseMax(-1.0 + kEps);
            Eigen::ArrayXd t = y_true.array().cwiseMax(-1.0 + kEps);
            return ((1.0 + p).log() - (1.0 + t).log()).square().mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = y_pred.array().cwiseMax(-1.0 + kEps);
            Eigen::ArrayXd t = y_true.array().cwiseMax(-1.0 + kEps);
            return (2.0 * ((1.0 + p).log() - (1.0 + t).log()) / (1.0 + p)).matrix();
        },
        "MSLE"
    );
}

inline LossFunction Poisson() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd lambda = y_pred.array().cwiseMax(kEps);
            return (lambda - y_true.array() * lambda.log()).mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd lambda = y_pred.array().cwiseMax(kEps);
            return (1.0 - (y_true.array() / lambda)).matrix();
        },
        "Poisson"
    );
}

inline LossFunction KLDivergence() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = clamp_prob(y_pred);
            Eigen::ArrayXd t = y_true.array().cwiseMax(kEps);
            return (t * (t / p).log()).mean();
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = clamp_prob(y_pred);
            Eigen::ArrayXd t = y_true.array().cwiseMax(kEps);
            return -(t / p).matrix();
        },
        "KLDivergence"
    );
}

// --- Categorical/Vector Functions ---

inline Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits) {
    if (logits.size() == 0) return Eigen::MatrixXd();
    double max_logit = logits.maxCoeff();
    Eigen::ArrayXd exps = (logits.array() - max_logit).exp();
    return (exps / exps.sum()).matrix();
}

inline LossFunction CategoricalCrossEntropy() {
    return LossFunction(
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            if (y_pred.size() != y_true.size()) throw std::invalid_argument("CCE size mismatch.");
            Eigen::ArrayXd p = clamp_prob(y_pred);
            return -(y_true.array() * p.log()).sum(); // CCE typically sums over classes
        },
        [](const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
            Eigen::ArrayXd p = clamp_prob(y_pred);
            return -(y_true.array() / p).matrix();
        },
        "CategoricalCrossEntropy"
    );
}

// Fused function to save calculating the Jacobian during backprop
// Update the fused loss for batch processing
inline LossFunction SoftmaxCrossEntropy() {
    return LossFunction(
        [](const Eigen::MatrixXd& logits, const Eigen::MatrixXd& y_true) {
            if (logits.size() != y_true.size()) throw std::invalid_argument("Size mismatch.");
            
            // Get probabilities using our new batch-compliant Softmax
            Eigen::MatrixXd probs = clamp_prob(activations::Softmax().forward(logits));
            
            // 1. Calculate CCE for each image independently
            // (y_true * log(probs)) gives a [Batch x 10] matrix. 
            // .rowwise().sum() squashes it to a [Batch x 1] vector of individual losses
            Eigen::VectorXd sample_losses = -(y_true.array() * probs.array().log()).matrix().rowwise().sum();
            
            // 2. Return the mean loss across the entire batch
            return sample_losses.mean();
        },
        [](const Eigen::MatrixXd& logits, const Eigen::MatrixXd& y_true) {
            // Derivative remains beautifully simple: Preds - True (now natively 2D)
            return activations::Softmax().forward(logits) - y_true;
        },
        "SoftmaxCrossEntropy"
    );
}

} // namespace losses

#endif // LOSS_FUNC_H