#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <png.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include "cnpy.h"

// Configurable parameters
const size_t START_SAMPLE = 0;         // Start sample index
const size_t END_SAMPLE = 22000;        // End sample index (exclusive)
const double THRESHOLD = 0.000001;          // Recurrence threshold
//const std::string CHANNEL1_NAME = "FP1-F7"; // First channel
const std::string CHANNEL1_NAME = "F7-T7"; // First channel
const std::string CHANNEL2_NAME = "F7-T7"; // Second channel (same for diagonal)
const size_t MIN_LINE_LENGTH = 2;      // Minimum length for diagonal lines (for DET, L, L_max)

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

// Compute cross-recurrence matrix and RQA metrics
void compute_crp(const std::vector<double>& data1, const std::vector<double>& data2,
                 size_t start, size_t end, double threshold, std::vector<char>& crp) {
    size_t n_samples = end - start;
    std::cout << "Computing CRP for range [" << start << ", " << end << "] (" << n_samples << " samples)" << std::endl;
    if (data1.size() < end || data2.size() < end) {
        throw std::runtime_error("Data arrays too small for range: data1.size=" + std::to_string(data1.size()) +
                                 ", data2.size=" + std::to_string(data2.size()) + ", end=" + std::to_string(end));
    }
    crp.resize(n_samples * n_samples);
    std::cout << "CRP matrix size: " << n_samples << " x " << n_samples << " (" << crp.size() * sizeof(char) / (1024.0 * 1024.0) << " MB)" << std::endl;

    // Compute recurrence matrix
    size_t count = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_samples; ++j) {
            double diff = std::abs(data1[start + i] - data2[start + j]);
            crp[i * n_samples + j] = (diff <= threshold) ? 1 : 0;
            if (crp[i * n_samples + j]) count++;
        }
        if (n_samples >= 10 && (i + 1) % (n_samples / 10) == 0) {
            std::cout << "Processed " << (i + 1) << "/" << n_samples << " rows (" << (100.0 * (i + 1) / n_samples) << "%)" << std::endl;
        }
    }
    double rr = 100.0 * count / (n_samples * n_samples);
    std::cout << "Recurrence Rate (RR): " << count << " recurrent points (" << rr << "% density)" << std::endl;

    // Compute diagonal line statistics (DET, L, L_max)
    std::vector<size_t> diag_histogram(n_samples + 1, 0); // Histogram of diagonal line lengths
    size_t diag_points = 0; // Total points in diagonal lines >= MIN_LINE_LENGTH
    size_t num_diag_lines = 0; // Number of diagonal lines >= MIN_LINE_LENGTH
    size_t l_max = 0; // Longest diagonal line

    // For same channel, exclude main diagonal (i == j) from DET, L, L_max
    bool same_channel = (&data1 == &data2);

    for (size_t d = (same_channel ? 1 : 0); d < n_samples; ++d) { // Start at d=1 for same channel
        for (size_t i = 0; i < n_samples - d; ++i) {
            size_t j = i + d; // Diagonal d: points (i, i+d)
            if (j >= n_samples) continue;
            size_t length = 0;
            while (i + length < n_samples && j + length < n_samples &&
                   crp[(i + length) * n_samples + (j + length)]) {
                ++length;
            }
            if (length >= MIN_LINE_LENGTH) {
                diag_histogram[length]++;
                diag_points += length;
                num_diag_lines++;
                l_max = std::max(l_max, length);
            }
            // Check negative diagonal for cross-CRP (not needed for self-CRP)
            if (!same_channel && i >= d) {
                j = i - d; // Diagonal -d: points (i, i-d)
                length = 0;
                while (i + length < n_samples && j + length < n_samples &&
                       crp[(i + length) * n_samples + (j + length)]) {
                    ++length;
                }
                if (length >= MIN_LINE_LENGTH) {
                    diag_histogram[length]++;
                    diag_points += length;
                    num_diag_lines++;
                    l_max = std::max(l_max, length);
                }
            }
        }
    }

    // Compute DET, L, L_max
    double det = count > 0 ? 100.0 * diag_points / count : 0.0;
    double l_avg = num_diag_lines > 0 ? static_cast<double>(diag_points) / num_diag_lines : 0.0;
    std::cout << "Determinism (DET): " << det << "% (" << diag_points << " points in diagonal lines >= " << MIN_LINE_LENGTH << ")" << std::endl;
    std::cout << "Average Diagonal Line Length (L): " << l_avg << std::endl;
    std::cout << "Longest Diagonal Line (L_max): " << l_max << std::endl;

    // Verify diagonal for same channel
    if (same_channel) {
        bool diagonal_valid = true;
        for (size_t i = 0; i < n_samples; ++i) {
            if (!crp[i * n_samples + i]) {
                diagonal_valid = false;
                std::cerr << "Warning: Diagonal point (" << i << ", " << i << ") is not set" << std::endl;
            }
        }
        std::cout << "Diagonal check: " << (diagonal_valid ? "Valid (all diagonal points set)" : "Invalid") << std::endl;
    }
}

// Save plot as PNG (unchanged from your version)
void save_png(Display* display, Window window, const std::string& filename, int width, int height) {
    std::cout << "Saving plot to " << filename << std::endl;
    XImage* image = XGetImage(display, window, 0, 0, width, height, AllPlanes, ZPixmap);
    if (!image) {
        throw std::runtime_error("Failed to get XImage");
    }

    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        XDestroyImage(image);
        throw std::runtime_error("Cannot open file " + filename);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info) {
        fclose(fp);
        XDestroyImage(image);
        throw std::runtime_error("PNG initialization failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        XDestroyImage(image);
        throw std::runtime_error("Error during PNG creation");
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    std::vector<png_byte> row(3 * width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned long pixel = XGetPixel(image, x, y);
            row[x * 3 + 0] = (pixel >> 16) & 0xff; // Red
            row[x * 3 + 1] = (pixel >> 8) & 0xff;  // Green
            row[x * 3 + 2] = pixel & 0xff;         // Blue
        }
        png_write_row(png, row.data());
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    XDestroyImage(image);
    std::cout << "Successfully saved plot to " << filename << std::endl;
}

int main() {
    try {
        // File paths
        const std::string data_file = "../data/p11_06_filtered.npy";
        const std::string metadata_file = "../data/p11_06_metadata.npy";
        const std::string output_png = "crp_plot.png";

        std::cout << "Starting CRP computation for " << data_file << std::endl;

        // Load .npy files using cnpy
        std::cout << "Loading data file: " << data_file << std::endl;
        cnpy::NpyArray data_array = cnpy::npy_load(data_file);
        if (data_array.shape.size() != 2) {
            throw std::runtime_error("Invalid data shape in " + data_file);
        }

        size_t n_channels = data_array.shape[0];
        size_t n_samples = data_array.shape[1];
        std::cout << "Data loaded: " << n_channels << " channels, " << n_samples << " samples" << std::endl;

        if (END_SAMPLE > n_samples || START_SAMPLE >= END_SAMPLE) {
            throw std::runtime_error("Invalid sample range: start=" + std::to_string(START_SAMPLE) +
                                     ", end=" + std::to_string(END_SAMPLE) +
                                     ", n_samples=" + std::to_string(n_samples));
        }

        std::vector<double> data(data_array.data<double>(), data_array.data<double>() + n_channels * n_samples);
        std::cout << "Data copied to vector: " << data.size() << " elements" << std::endl;

        // Load metadata
        std::cout << "Loading metadata file: " << metadata_file << std::endl;
        cnpy::NpyArray metadata_array = cnpy::npy_load(metadata_file);
        std::vector<std::string> channel_names;
        std::ifstream ch_file("channels.txt");
        if (!ch_file.is_open()) {
            std::cerr << "Warning: channels.txt not found, using default channel names" << std::endl;
            channel_names = {"FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                             "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                             "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"};
        } else {
            std::string line;
            while (std::getline(ch_file, line)) {
                channel_names.push_back(line);
            }
            ch_file.close();
        }
        std::cout << "Available channels: ";
        for (const auto& ch : channel_names) std::cout << ch << " ";
        std::cout << std::endl;

        // Find channel indices
        int ch1_idx = -1, ch2_idx = -1;
        for (size_t i = 0; i < channel_names.size(); ++i) {
            if (channel_names[i] == CHANNEL1_NAME) ch1_idx = i;
            if (channel_names[i] == CHANNEL2_NAME) ch2_idx = i;
        }
        if (ch1_idx == -1 || ch2_idx == -1) {
            throw std::runtime_error("Channel not found: " + CHANNEL1_NAME + " or " + CHANNEL2_NAME);
        }
        std::cout << "Selected channel 1: " << CHANNEL1_NAME << " (index " << ch1_idx << ")" << std::endl;
        std::cout << "Selected channel 2: " << CHANNEL2_NAME << " (index " << ch2_idx << ")" << std::endl;

        // Extract channels
        std::vector<double> channel1(data.begin() + ch1_idx * n_samples, data.begin() + (ch1_idx + 1) * n_samples);
        std::vector<double> channel2(data.begin() + ch2_idx * n_samples, data.begin() + (ch2_idx + 1) * n_samples);
        std::cout << "Extracted channel 1 (" << CHANNEL1_NAME << "): " << channel1.size() << " samples" << std::endl;
        std::cout << "Extracted channel 2 (" << CHANNEL2_NAME << "): " << channel2.size() << " samples" << std::endl;

        // Print sample values
        std::cout << "Sample data (channel 1, " << CHANNEL1_NAME << "): ";
        for (size_t i = START_SAMPLE; i < std::min(START_SAMPLE + 5, END_SAMPLE); ++i) {
            std::cout << channel1[i] << " ";
        }   
        std::cout << std::endl;
        std::cout << "Sample data (channel 2, " << CHANNEL2_NAME << "): ";
        for (size_t i = START_SAMPLE; i < std::min(START_SAMPLE + 5, END_SAMPLE); ++i) {
            std::cout << channel2[i] << " ";
        }
        std::cout << std::endl;

        // Compute cross-recurrence matrix
        std::vector<char> crp;
        std::cout << "Entering compute_crp()" << std::endl;
        compute_crp(channel1, channel2, START_SAMPLE, END_SAMPLE, THRESHOLD, crp);

        // Initialize X11
        std::cout << "Initializing X11 display" << std::endl;
        Display* display = XOpenDisplay(nullptr);
        if (!display) {
            throw std::runtime_error("Cannot open X11 display");
        }

        int screen = DefaultScreen(display);
        Window window = XCreateSimpleWindow(display, RootWindow(display, screen), 10, 10,
                                           WINDOW_WIDTH, WINDOW_HEIGHT, 1,
                                           BlackPixel(display, screen), WhitePixel(display, screen));
        XSelectInput(display, window, ExposureMask | KeyPressMask);
        XMapWindow(display, window);
        std::cout << "X11 window created (" << WINDOW_WIDTH << "x" << WINDOW_HEIGHT << ")" << std::endl;

        GC gc = XCreateGC(display, window, 0, nullptr);
        XSetForeground(display, gc, BlackPixel(display, screen));

        // Main event loop
        std::cout << "Entering X11 event loop (press any key to save and exit)" << std::endl;
        bool running = true;
	std::cout << "plotting start" << std::endl;
        while (running) {
            XEvent event;
            XNextEvent(display, &event);
            if (event.type == Expose) {
                std::cout << "Drawing CRP in X11 window" << std::endl;
                size_t n_samples = END_SAMPLE - START_SAMPLE;
                for (size_t i = 0; i < n_samples; ++i) {
                    for (size_t j = 0; j < n_samples; ++j) {
                        if (crp[i * n_samples + j]) {
                            int x = (j * WINDOW_WIDTH) / n_samples;
                            int y = (i * WINDOW_HEIGHT) / n_samples;
                            XDrawPoint(display, window, gc, x, y);
                        }
                    }
                }
                std::cout << "Drawing channel labels" << std::endl;
                XDrawString(display, window, gc, 10, WINDOW_HEIGHT - 10, CHANNEL1_NAME.c_str(), CHANNEL1_NAME.length());
                XDrawString(display, window, gc, 10, 20, CHANNEL2_NAME.c_str(), CHANNEL2_NAME.length());
            }
            else if (event.type == KeyPress) {
                std::cout << "Key pressed, saving plot and exiting" << std::endl;
                save_png(display, window, output_png, WINDOW_WIDTH, WINDOW_HEIGHT);
                running = false;
            }
        }

        // Cleanup
        std::cout << "Cleaning up X11 resources" << std::endl;
        XFreeGC(display, gc);
        XDestroyWindow(display, window);
        XCloseDisplay(display);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Program completed successfully" << std::endl;
    return 0;
}
