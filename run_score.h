#ifndef RUN_SCORES
#define RUN_SCORES

#include<vector>
#include <stdexcept>

double r2_score(const std::vector<double>& y_true,
                const std::vector<double>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    double mean = 0.0;
    for (double v : y_true) mean += v;
    mean /= y_true.size();

    double ss_res = 0.0, ss_tot = 0.0;

    for (size_t i = 0; i < y_true.size(); i++) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
    }

    return 1.0 - (ss_res / ss_tot);
}

double r2_score(const std::vector<std::vector<double>>& y_true,
                const std::vector<std::vector<double>>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    double total_r2 = 0.0;
    int outputs = y_true[0].size();

    for (int j = 0; j < outputs; j++) {
        std::vector<double> yt, yp;

        for (size_t i = 0; i < y_true.size(); i++) {
            yt.push_back(y_true[i][j]);
            yp.push_back(y_pred[i][j]);
        }

        total_r2 += r2_score(yt, yp);
    }

    return total_r2 / outputs;
}

double accuracy_score(const std::vector<double>& y_true,
                      const std::vector<double>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    int correct = 0;

    for (size_t i = 0; i < y_true.size(); i++) {
        int pred = (y_pred[i] >= 0.5) ? 1 : 0;
        int actual = (y_true[i] >= 0.5) ? 1 : 0;

        if (pred == actual) correct++;
    }

    return (double)correct / y_true.size();
}

double accuracy_score(const std::vector<std::vector<double>>& y_true,
                      const std::vector<std::vector<double>>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    int correct = 0;
    int total = 0;

    for (size_t i = 0; i < y_true.size(); i++) {
        for (size_t j = 0; j < y_true[i].size(); j++) {
            int pred = (y_pred[i][j] >= 0.5) ? 1 : 0;
            int actual = (y_true[i][j] >= 0.5) ? 1 : 0;

            if (pred == actual) correct++;
            total++;
        }
    }

    return (double)correct / total;
}

#endif
