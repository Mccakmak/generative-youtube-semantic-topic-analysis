import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

def filter_emotions(df, attr1, attr2, rel1, rel2):
    return df.loc[~((df[attr1] == rel1) & (df[attr2] == rel2))]

def merge(df_orig, df_filtered):
    df_orig_filtered = df_orig[df_orig['depth'] != 0]
    updated_df = pd.concat([df_orig_filtered, df_filtered])
    return updated_df

def clean_subsequent_depth(df):
    root_video_ids = df[df['depth'] == 0]['root_video_id'].unique()
    filtered_df = df[df['root_video_id'].isin(root_video_ids)]
    return filtered_df

def plot_emotion_distribution(df, file_name):
    # Combine emotions from title and transcription
    df['emotions_combined'] = df[['emotion_title', 'emotion_transcription']].apply(lambda row: row.dropna().values.tolist(), axis=1)
    df = df.explode('emotions_combined')

    # Filter out non-integer values
    df = df[df['emotions_combined'].apply(lambda x: str(x).isdigit())]
    df['emotions_combined'] = df['emotions_combined'].astype(int)  # Ensure emotions are integers

    # Group by depth and emotions, then calculate the ratio for each emotion
    emotions_counts = df.groupby(['depth', 'emotions_combined']).size().unstack(fill_value=0)
    emotions_ratios = emotions_counts.div(emotions_counts.sum(axis=1), axis=0).reset_index()

    emotions_ratios.to_csv("output/" + file_name + "_emotions_probabilities.csv", index=False)

    # Ensure the data is sorted by depth
    emotions_ratios = emotions_ratios.sort_values('depth')
    emotions_ratios = emotions_ratios.reset_index(drop=True)

    # Plot the emotions distribution
    plt.figure(figsize=(12, 10))

    # Define colors and labels for each emotion
    colors = {
        0: 'orange',       # Joy/Happiness
        1: 'royalblue',  # Sadness
        2: 'red',          # Anger
        3: 'forestgreen',    # Neutral
        4: 'indigo'        # Fear
    }

    emotion_labels = {
        0: 'Joy/Happiness',
        1: 'Sadness',
        2: 'Anger',
        3: 'Neutral',
        4: 'Fear'
    }

    less_visible_depths = [13, 18, 23, 28, 33, 38, 43, 48]

    # Prepare legend handles
    legend_handles = []

    # Plot the ratios and moving averages for each emotion
    for emotion in emotions_ratios.columns[1:]:
        emotion_int = int(emotion)
        color = colors[emotion_int]
        label = emotion_labels[emotion_int]
        x = emotions_ratios['depth'].values
        y = emotions_ratios[emotion].values

        # Moving average with window size 5
        moving_avg = emotions_ratios[emotion].rolling(window=5, center=True).mean()
        (line,) = plt.plot(emotions_ratios['depth'], moving_avg, linestyle='--', label=f'{label} Moving Avg', color=color)
        legend_handles.append(line)

        # Prepare line segments with adjusted transparency
        segments = []
        segment_colors = []
        for i in range(len(x)-1):
            x0, y0 = x[i], y[i]
            x1, y1 = x[i+1], y[i+1]
            segments.append([(x0, y0), (x1, y1)])

            # Determine alpha
            if x0 in less_visible_depths or x1 in less_visible_depths:
                alpha = 0.3
            else:
                alpha = 1.0

            # Get RGBA color with adjusted alpha
            color_rgba = mcolors.to_rgba(color)
            color_with_alpha = list(color_rgba)
            color_with_alpha[3] = alpha
            segment_colors.append(tuple(color_with_alpha))

        # Create LineCollection
        lc = LineCollection(segments, colors=segment_colors, linewidths=2)
        plt.gca().add_collection(lc)

        # Plot markers with adjusted transparency
        marker_alphas = [0.3 if xi in less_visible_depths else 1.0 for xi in x]
        # Create colors with adjusted alpha
        marker_colors = []
        for alpha in marker_alphas:
            color_rgba = mcolors.to_rgba(color)
            color_with_alpha = list(color_rgba)
            color_with_alpha[3] = alpha
            marker_colors.append(tuple(color_with_alpha))

        scatter = plt.scatter(x, y, color=marker_colors, s=50, zorder=3)
        # Add the original data to legend only once per emotion
        if i == 0:
            legend_handles.append(scatter)

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=32)
    plt.ylabel('Emotion Distribution Ratios', fontsize=32)
    # plt.title('Topic Distribution by Depth', fontsize=14)

    # Adjust legend to specific positions
    #plt.legend(legend_handles, [h.get_label() for h in legend_handles], fontsize=16, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Adjust tick font size
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_plot.pdf')

if __name__ == "__main__":
    file_name = "scs_recommended_D50_T60_common_w0_video_final_transcript_translated_emotions"
    df = pd.read_csv("output/" + file_name + ".csv")

    df['emotion_title'] = df['emotion_title'].astype(int)
    df['emotion_transcription'] = df['emotion_transcription'].astype(int)

    # Filter depth 0, remove 0 relevancies
    df_depth_0 = df[df["depth"] == 0]

    df_filtered_depth_0 = filter_emotions(df_depth_0, attr1="emotion_title", attr2="emotion_transcription", rel1=0, rel2=0)

    df = merge(df, df_filtered_depth_0)
    df_filtered = clean_subsequent_depth(df)

    plot_emotion_distribution(df_filtered, file_name)
