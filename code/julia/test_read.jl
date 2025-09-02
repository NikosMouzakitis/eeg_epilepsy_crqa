#using CondaPkg
#CondaPkg.add("numpy")
using NPZ
using PythonCall
using Plots

# Define paths to the .npy files
data_file = "../../data/p11_06_filtered.npy"
metadata_file = "../../data/p11_06_metadata.npy"

# Check if files exist
if !isfile(data_file)
    error("Data file not found: $data_file")
end
if !isfile(metadata_file)
    error("Metadata file not found: $metadata_file")
end

# Read the filtered EEG data using NPZ.jl
eeg_data = npzread(data_file)

# Read the metadata using PythonCall.jl
np = pyimport("numpy")  # Import Python's NumPy module
metadata = np.load(metadata_file, allow_pickle=true)  # Allow pickle to handle object arrays
metadata_dict = pyconvert(Dict, metadata.item())  # Convert Python dict to Julia Dict

# Extract metadata fields
channels = pyconvert(Vector{String}, metadata_dict["channels"])  # Convert to Julia String array
sfreq = pyconvert(Float64, metadata_dict["sfreq"])  # Convert to Julia Float64
times = pyconvert(Vector{Float64}, metadata_dict["times"])  # Convert to Julia Float64 array

# Print information to verify
println("EEG Data Shape: ", size(eeg_data))
println("Channels: ", channels)
println("Sampling Frequency: ", sfreq)
println("Number of Time Points: ", length(times))

# Example: Access the data for a specific channel
first_channel_data = eeg_data[1, :]
println("First Channel Data (first 10 samples): ", first_channel_data[1:10])

# Optional: Plot the first channel's data (requires Plots.jl)
using Plots
plot(times, first_channel_data, label=channels[1], title="EEG Signal for $(channels[1])", xlabel="Time (s)", ylabel="Amplitude")
