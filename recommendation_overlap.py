import pandas as pd
import matplotlib.pyplot as plt

# Load the data (replace the file paths with your actual paths)
file_3 = pd.read_csv('output/scs_recommended_D50_T3_common_w0_video_final_transcript_translated_emotions.csv')
file_15 = pd.read_csv('output/scs_recommended_D50_T15_common_w0_video_final_transcript_translated_emotions.csv')
file_60 = pd.read_csv('output/scs_recommended_D50_T60_common_w0_video_final_transcript_translated_emotions.csv')

# Extract the necessary columns
file_15_videos = file_15[['video_id', 'depth']]
file_3_videos = file_3[['video_id', 'depth']]
file_60_videos = file_60[['video_id', 'depth']]

# Get unique depths
unique_depths = sorted(set(file_15_videos['depth']).union(set(file_3_videos['depth'])).union(set(file_60_videos['depth'])))

# Initialize a dictionary to store overlap results
overlap_results = {'depth': [], '3-15 seconds': [], '3-60 seconds': [], '15-60 seconds': []}

# Compare the video_id overlap for each depth
for depth in unique_depths:
    # Get video_ids for each file at the given depth
    videos_15 = set(file_15_videos[file_15_videos['depth'] == depth]['video_id'])
    videos_3 = set(file_3_videos[file_3_videos['depth'] == depth]['video_id'])
    videos_60 = set(file_60_videos[file_60_videos['depth'] == depth]['video_id'])

    # Calculate overlap
    overlap_15_3 = len(videos_15.intersection(videos_3))
    overlap_15_60 = len(videos_15.intersection(videos_60))
    overlap_3_60 = len(videos_3.intersection(videos_60))

    # Store results
    overlap_results['depth'].append(depth)
    overlap_results['3-15 seconds'].append(overlap_15_3)
    overlap_results['3-60 seconds'].append(overlap_15_60)
    overlap_results['15-60 seconds'].append(overlap_3_60)

# Convert results to DataFrame
overlap_df = pd.DataFrame(overlap_results)

# Calculate the total videos for each depth in file_15
total_videos_per_depth = file_15_videos.groupby('depth')['video_id'].count().reindex(unique_depths, fill_value=0)

# Filter the DataFrame for specified depths
depths_to_include = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
filtered_overlap_df = overlap_df[overlap_df['depth'].isin(depths_to_include)]
filtered_total_videos = total_videos_per_depth[depths_to_include]

# Plotting the overlaps as a bar plot with total videos line graph
fig, ax1 = plt.subplots(figsize=(14, 8))

bar_width = 0.25
index = range(len(filtered_overlap_df))

# Bar plots for overlaps
ax1.bar(index, filtered_overlap_df['3-15 seconds'], bar_width, label='3-15 seconds')
ax1.bar([i + bar_width for i in index], filtered_overlap_df['3-60 seconds'], bar_width, label='3-60 seconds')
ax1.bar([i + 2 * bar_width for i in index], filtered_overlap_df['15-60 seconds'], bar_width, label='15-60 seconds')

ax1.set_xlabel('Depth')
ax1.set_ylabel('Number of Overlapping Video IDs')
ax1.set_title('Overlap of Video IDs at Different Depths')
ax1.set_xticks([i + bar_width for i in index])
ax1.set_xticklabels(filtered_overlap_df['depth'])
ax1.legend(loc='upper left')
ax1.grid(False)

# Secondary axis for total videos
ax2 = ax1.twinx()
ax2.plot(index, filtered_total_videos, marker='o', color='purple', label='Total Videos')

ax2.set_ylabel('Total Videos at Depth')
ax2.legend(loc='upper right')

plt.show()
