
#include <vector>
#include <random>
#include <string>

#include "model.h"
#include "run_score.h"

using namespace std;

#define FAST                     \
    ios::sync_with_stdio(false); \
    cin.tie(nullptr);            \
    cout.tie(nullptr);
int main()
{
    FAST;
    int samples = 1000;
    int features = 5;

    vector<vector<double>> X(samples, vector<double>(features));
    vector<vector<double>> Y(samples, vector<double>(5));

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dist(-10.0, 10.0);
    normal_distribution<double> noise(0.0, 0.5); // small noise

    for (int i = 0; i < samples; i++)
    {
        double x1 = dist(gen);
        double x2 = dist(gen);
        double x3 = dist(gen);
        double x4 = dist(gen);
        double x5 = dist(gen);

        X[i] = {x1, x2, x3, x4, x5};

        Y[i][0] = 2 * x1 - 3 * x2 + 0.5 * x3 + 4 * x4 - x5 + noise(gen);
        Y[i][1] = 1 * x1 + 3 * x2 + 5 * x3 + 4 * x4 - x5 + noise(gen);
        Y[i][2] = 20 * x1 - 3.5 * x2 + 2.5 * x3 + 8 * x4 - x5 + noise(gen);
        Y[i][3] = 6 * x1 + 3 * x2 + 0.5 * x3 + 4 * x4 - x5 + noise(gen);
        Y[i][4] = x1 - 3 * x2 + 10.5 * x3 + 10 * x4 - x5 + noise(gen);
    }
    auto program_start = chrono::high_resolution_clock::now();

    LossFunction loss = losses::MSE();
    nn::Optimizer opt = nn::Adam();
    nn::SequentialNetwork model(loss, opt, 1000);
    model.add_dense_layer(5,5,activations::Identity());
    model.fit(X, Y);
    auto preds = model.predict(X);

    auto program_end = chrono::high_resolution_clock::now();

    auto duration =
        chrono::duration_cast<chrono::seconds>(
            program_end - program_start);

    cout << "\nTotal execution time: "
         << duration.count() << " seconds\n";
    for (int i = 0; i < 10; i++)
    {
        cout << "X: ";
        for (auto v : X[i])
        {
            cout << v << " ";
        }
        for (int j = 0; j < 5; j++)
        {
            cout << " | True: " << Y[i][j]
                 << " | Pred: " << preds[i][j]
                 << endl;
        }

        cout << endl;
    }

    cout << "R2 score is " << r2_score(Y, preds) << endl;
    model.save_model("custom.h");
    model.summary();
    return 0;
}