import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

# Read the CSV file from the specified path
df = pd.read_csv('output/merged_Taiwan_YT_regular_no_symbols_video_translated_final_transcript_unique_filtered_gpt_relevance_emotion.csv')

# Emotion labels and colors, excluding 'Neutral' (which has code 3)
emotion_labels = {
    0: 'Happiness',
    1: 'Sadness',
    2: 'Anger',
    # 3: 'Neutral',  # Excluded
    4: 'Fear',
    5: 'Disgust',
    6: 'Surprise'
}

emotion_colors = {
    0: 'orange',
    1: 'blue',
    2: 'red',
    # 3: 'gray',  # Excluded
    4: 'black',
    5: 'green',
    6: 'pink'
}

# Columns to generate graphs for
emotion_columns = ['emotion_title', 'emotion_description', 'emotion_transcription']

# Font sizes
label_font_size = 20
tick_font_size = 20
legend_font_size = 18

# Loop through each emotion column
for emotion_col in emotion_columns:
    # Create a copy of the DataFrame to avoid modifying the original during cleaning
    df_clean = df.copy()

    # **Data Cleaning Steps**

    # 1. Handle invalid 'depth' values
    # Convert 'depth' to numeric, coerce errors to NaN
    df_clean['depth'] = pd.to_numeric(df_clean['depth'], errors='coerce')

    # Keep only rows where 'depth' is in [0, 1, 2, 3]
    df_clean = df_clean[df_clean['depth'].isin([0, 1, 2, 3])]

    # 2. Handle invalid emotion column values
    # Convert emotion column to numeric, coerce errors to NaN
    df_clean[emotion_col] = pd.to_numeric(df_clean[emotion_col], errors='coerce')

    # Drop any rows with NaN values in the emotion column or 'depth'
    df_clean = df_clean.dropna(subset=[emotion_col, 'depth'])

    # **Exclude 'Neutral' Emotion (code 3)**
    df_clean = df_clean[df_clean[emotion_col] != 3]

    # Keep only rows where emotion values are in the defined emotion labels
    df_clean = df_clean[df_clean[emotion_col].isin(emotion_labels.keys())]

    # **Proceed with the Analysis**

    # Group by 'depth' and the emotion column to count occurrences
    counts = df_clean.groupby(['depth', emotion_col]).size().reset_index(name='counts')

    # Get total counts per depth (excluding 'Neutral')
    total_counts = counts.groupby('depth')['counts'].sum().reset_index(name='total_counts')

    # Merge counts with total_counts
    merged = pd.merge(counts, total_counts, on='depth')

    # Calculate ratios
    merged['ratio'] = merged['counts'] / merged['total_counts']

    # Pivot the DataFrame to have emotions as columns and depths as rows
    pivot_df = merged.pivot(index='depth', columns=emotion_col, values='ratio')

    # Ensure all depths and emotions are represented
    depths = [0, 1, 2, 3]
    emotions = list(emotion_labels.keys())
    pivot_df = pivot_df.reindex(index=depths, columns=emotions, fill_value=0)

    # **Ensure that data for all depths and emotions is filled with zeros where missing**
    pivot_df = pivot_df.fillna(0)

    # Plotting
    plt.figure(figsize=(10, 8))

    for emotion in pivot_df.columns:
        plt.plot(
            pivot_df.index,
            pivot_df[emotion],
            label=emotion_labels[emotion],
            color=emotion_colors[emotion],
            #marker='o'  # Optional: Add markers to points
        )

    # Adjust font sizes
    plt.xlabel('Depth', fontsize=label_font_size)
    plt.ylabel('Ratio', fontsize=label_font_size)
    # Remove the title as per your request
    # plt.title(f'Emotion Trend Graph for {emotion_col} (Excluding Neutral)', fontsize=label_font_size)

    plt.xticks(depths, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.ylim(0, 1)
    plt.legend(fontsize=legend_font_size)
    plt.grid(False)  # Optional: Add grid lines

    # Save the figure as a PDF file with tight layout in the 'results' directory
    plt.savefig(f'results/{emotion_col}_emotion_no_symbol.pdf', bbox_inches='tight')

    # Clear the figure for the next plot
    plt.clf()

    # Optional: Display the plot
    # plt.show()
