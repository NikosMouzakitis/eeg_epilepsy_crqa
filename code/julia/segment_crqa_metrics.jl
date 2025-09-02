using DynamicalSystems
using RecurrenceAnalysis  # For rqa
using NPZ
using PythonCall
using Plots
using Statistics  # For std in threshold calculation
using CSV, DataFrames  # For saving results
plotlyjs()  # Use PlotlyJS backend for interactive plots

# Load EEG data
data_file = "../../data/p11_06_filtered.npy"
metadata_file = "../../data/p11_06_metadata.npy"

# Check if files exist
if !isfile(data_file) || !isfile(metadata_file)
    error("File not found: $data_file or $metadata_file")
end

# Read EEG data and metadata
eeg_data = npzread(data_file)
np = pyimport("numpy")
metadata = np.load(metadata_file, allow_pickle=true)
metadata_dict = pyconvert(Dict, metadata.item())
channels = pyconvert(Vector{String}, metadata_dict["channels"])
sfreq = pyconvert(Float64, metadata_dict["sfreq"])
times = pyconvert(Vector{Float64}, metadata_dict["times"])

# Select two EEG channels for CRQA (e.g., FP1-F7 and F7-T7)
ch1_idx = findfirst(x -> x == "FP1-F7", channels)
ch2_idx = findfirst(x -> x == "F7-T7", channels)
if isnothing(ch1_idx) || isnothing(ch2_idx)
    error("Channels FP1-F7 or F7-T7 not found")
end
signal1 = eeg_data[ch1_idx, :]
signal2 = eeg_data[ch2_idx, :]

# Segment parameters
segment_length_sec = 5.0  # 5 seconds
samples_per_segment = Int(sfreq * segment_length_sec)  # Number of samples in 5 seconds
n_samples = length(signal1)
n_segments = div(n_samples, samples_per_segment)  # Number of full segments

# CRQA parameters
τ = 10  # Delay (in samples, e.g., 10 samples ≈ 40ms at 250Hz)
dim = 3  # Embedding dimension

# Store CRQA results for each segment
crqa_results = Dict(
    "segment_times" => Float64[],
    "rr" => Float64[],
    "det" => Float64[],
    "avgdiag" => Float64[],
    "entropy" => Float64[]
)

# Perform CRQA for each segment
for i in 1:n_segments
    # Extract segment
    start_idx = (i - 1) * samples_per_segment + 1
    end_idx = i * samples_per_segment
    segment1 = signal1[start_idx:end_idx]
    segment2 = signal2[start_idx:end_idx]
    
    # Skip if segment is too short
    if length(segment1) < dim * τ
        println("Segment $i too short for embedding, skipping")
        continue
    end
    
    # Compute recurrence threshold per segment
    ε = 0.01
    #ε = 0.3 * std(segment1)  # Increased to 30% for better recurrence
    
    # Embed the signals
    embedded1 = embed(segment1, dim, τ)
    embedded2 = embed(segment2, dim, τ)
    
    # Compute cross-recurrence plot
    crp = CrossRecurrenceMatrix(embedded1, embedded2, ε)
    
    # Compute CRQA metrics
    metrics = rqa(crp; theiler=1, minline=2)
    
    # Handle NaN entropy
    entropy = isnan(metrics[:ENTR]) ? 0.0 : metrics[:ENTR]
    
    # Store results
    segment_time = times[start_idx]  # Start time of segment
    push!(crqa_results["segment_times"], segment_time)
    push!(crqa_results["rr"], metrics[:RR])
    push!(crqa_results["det"], metrics[:DET])
    push!(crqa_results["avgdiag"], metrics[:L])
    push!(crqa_results["entropy"], entropy)
    
    println("Segment $i (t=$(segment_time)s): RR=$(metrics[:RR]), DET=$(metrics[:DET]), AvgDiag=$(metrics[:L]), Entropy=$entropy")
end

# Save results to CSV
df = DataFrame(crqa_results)
CSV.write("crqa_results.csv", df)

# Plot CRQA metrics over time
if !isempty(crqa_results["segment_times"])
    plot_layout = @layout [a; b; c; d]
    p1 = plot(crqa_results["segment_times"], crqa_results["rr"], label="Recurrence Rate", title="CRQA Metrics Over Time", ylabel="RR")
    p2 = plot(crqa_results["segment_times"], crqa_results["det"], label="Determinism", ylabel="DET")
    p3 = plot(crqa_results["segment_times"], crqa_results["avgdiag"], label="Avg Diagonal Line", ylabel="AvgDiag")
    p4 = plot(crqa_results["segment_times"], crqa_results["entropy"], label="Entropy", xlabel="Time (s)", ylabel="Entropy")
    metrics_plot = plot(p1, p2, p3, p4, layout=plot_layout)
    display(metrics_plot)
    savefig(metrics_plot, "crqa_metrics.png")
else
    println("No CRQA results to plot")
end

# Plot cross-recurrence plot for the first segment
first_segment1 = signal1[1:samples_per_segment]
first_segment2 = signal2[1:samples_per_segment]
if length(first_segment1) >= dim * τ
    embedded1 = embed(first_segment1, dim, τ)
    embedded2 = embed(first_segment2, dim, τ)
    ε_first = 0.3 * std(first_segment1)
    crp = CrossRecurrenceMatrix(embedded1, embedded2, ε_first)
    crp_plot = heatmap(crp, title="Cross-Recurrence Plot: $(channels[ch1_idx]) vs $(channels[ch2_idx]) (First 5s)",
                       xlabel="Time (samples) in $(channels[ch2_idx])", ylabel="Time (samples) in $(channels[ch1_idx])")
    display(crp_plot)
    savefig(crp_plot, "crp_first_segment.png")
else
    println("First segment too short for CRQA plot")
end
