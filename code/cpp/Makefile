# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -g -I$(HOME)/implementations/eeg_epilepsy_crqa/libs/cnpy-master
LDFLAGS = -L/usr/local/lib -lX11 -lpng -lz -lcnpy -Wl,-rpath=/usr/local/lib

# Target executable
TARGET = 4_cpp_visualization_metrics

# Source file
SRC=4_cpp_visualization_metrics.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS)
# Run the program
run: $(TARGET)
	./$(TARGET)
# Clean up
clean:
	rm -f a.out

.PHONY: all run clean



