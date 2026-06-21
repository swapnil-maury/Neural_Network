# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -I . -O3 -march=native -fopenmp

# Source files and Target executable
SRCS = $(wildcard mnist*.cpp)
TARGET = a.out

# Default target: just compile the code
all: $(TARGET)

# Rule to compile the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Rule to execute with unlimited stack size (prevents segfaults on large models)
run: $(TARGET)
	ulimit -s unlimited && ./$(TARGET)

# Rule to force a fresh recompile and run (Useful for your retraining workflow)
retrain: clean run

# Rule to download and setup the Eigen library locally
setup_eigen:
	@echo "Downloading and extracting Eigen 3.4.0..."
	curl -O https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
	tar -xzf eigen-3.4.0.tar.gz
	mv eigen-3.4.0/Eigen ./Eigen
	rm -rf eigen-3.4.0 eigen-3.4.0.tar.gz
	@echo "Eigen setup complete. Core headers inside ./Eigen:"
	ls ./Eigen

# Clean up the compiled binary
clean:
	rm -f $(TARGET)

# Prevent Make from confusingn targets with actual file names
.PHONY: all run retrain setup_eigen clea