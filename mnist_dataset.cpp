#include <fstream>
#include <iostream>
#include <vector>

#include "model.h"
#include "run_score.h"

// Reads X data (Images)
std::vector<std::vector<double>> load_X_bin(const std::string& filename, int samples, int features = 784) {
    std::vector<std::vector<double>> data(samples, std::vector<double>(features));
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Could not open " << filename << "\n"; return data; }
    
    for (int i = 0; i < samples; ++i) {
        in.read(reinterpret_cast<char*>(data[i].data()), features * sizeof(double));
    }
    return data;
}

// Reads Y data (One-Hot Labels)
std::vector<std::vector<double>> load_Y_bin(const std::string& filename, int samples, int classes = 10) {
    std::vector<std::vector<double>> data(samples, std::vector<double>(classes));
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Could not open " << filename << "\n"; return data; }
    
    for (int i = 0; i < samples; ++i) {
        in.read(reinterpret_cast<char*>(data[i].data()), classes * sizeof(double));
    }
    return data;
}

int main() {
    int samples = 10000; 
    std::cout << "Loading data...\n";
    auto X_train = load_X_bin("X_train.bin", samples, 784);
    auto Y_train = load_Y_bin("Y_train.bin", samples, 10);
    std::cout << "Data loaded!\n";

    // 1. Build the MNIST Architecture (Namespaces fixed!)
    LossFunction loss = losses::MSE(); 
    nn::Optimizer opt = nn::Adam(0.001); 
    nn::SequentialNetwork model(loss, opt, 784); 

    ActivationFunction relu = activations::ReLU(); 
    
    // Hidden Layers (Dropout pointer fixed!)
    model.add_dense_layer(784, 128, relu);
    model.add_layer(nn::DropoutLayer(0.2)); 
    model.add_dense_layer(128, 64, relu);
    
    // Output Layer (10 digits)
    model.add_dense_layer(64, 10, activations::Identity()); 

    // 2. Train the Model
    std::cout << "Starting training...\n";
    model.set_batch_size(64);
    model.set_epochs(10);
    model.fit(X_train, Y_train);

    // 3. Predict and Score
    std::cout << "Making predictions...\n";
    auto preds = model.predict(X_train);

    // Get true classification accuracy
    double acc = accuracy_score_multiclass(Y_train, preds);
    std::cout << "\n========================================\n";
    std::cout << "Training Accuracy: " << acc * 100.0 << "%\n";
    std::cout << "========================================\n";

    // Generate and print the confusion matrix
    auto conf_matrix = confusion_matrix(Y_train, preds);
    print_confusion_matrix(conf_matrix);

    // 4. Summarize and Save as RAW TEXT!
    std::cout << "\nSaving Model...\n";
    model.save_model("mnist_model.txt");
    model.summary();

    return 0;
}