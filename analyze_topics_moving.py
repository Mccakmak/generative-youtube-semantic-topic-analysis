import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

def filter_topics(df, attr1, attr2, rel1, rel2):
    return df.loc[~((df[attr1] == rel1) & (df[attr2] == rel2))]

def merge(df_orig, df_filtered):
    df_orig_filtered = df_orig[df_orig['depth'] != 0]
    updated_df = pd.concat([df_orig_filtered, df_filtered])
    return updated_df

def clean_subsequent_depth(df):
    root_video_ids = df[df['depth'] == 0]['root_video_id'].unique()
    filtered_df = df[df['root_video_id'].isin(root_video_ids)]
    return filtered_df

def plot_topic_distribution(df, file_name):
    # Combine topics from title and transcription
    df['topics_combined'] = df[['topics_title', 'topics_transcription']].apply(lambda row: row.dropna().values.tolist(),
                                                                               axis=1)
    df = df.explode('topics_combined')

    # Filter out non-integer values
    df = df[df['topics_combined'].apply(lambda x: str(x).isdigit())]
    df['topics_combined'] = df['topics_combined'].astype(int)  # Ensure topics are integers

    # Group by depth and topic, then calculate the ratio for each topic
    topic_counts = df.groupby(['depth', 'topics_combined']).size().unstack(fill_value=0)
    topic_ratios = topic_counts.div(topic_counts.sum(axis=1), axis=0).reset_index()

    topic_ratios.to_csv("output/" + file_name + "_topic_probabilities.csv", index=False)

    # Ensure the data is sorted by depth
    topic_ratios = topic_ratios.sort_values('depth').reset_index(drop=True)

    # Define the depths where data points and lines should be less visible
    less_visible_depths = [13, 18, 23, 28, 33, 38, 43, 48]

    # Plot the topic distribution
    plt.figure(figsize=(12, 10))

    # Define colors and labels for each topic
    colors = {
        0: 'royalblue',       # Politics
        1: 'forestgreen',  # Non-Entertainment
        2: 'orange',    # Entertainment
        # Add more colors if needed
    }

    topic_labels = {
        0: 'Politics',
        1: 'Non-Entertainment',
        2: 'Entertainment',
        # Add more labels if needed
    }

    # Prepare legend handles
    legend_handles = []

    # Plot the ratios and moving averages for each topic
    for topic in topic_ratios.columns[1:]:
        topic_int = int(topic)
        color = colors.get(topic_int, f'C{topic_int}')  # Default color if not specified
        label = topic_labels.get(topic_int, f'Topic {topic_int}')
        x = topic_ratios['depth'].values
        y = topic_ratios[topic].values

        # Moving average with window size 5
        moving_avg = topic_ratios[topic].rolling(window=5, center=True).mean()
        (line,) = plt.plot(x, moving_avg, linestyle='--', label=f'{label} Moving Avg', color=color)
        legend_handles.append(line)

        # Prepare line segments with adjusted transparency
        segments = []
        segment_colors = []
        for i in range(len(x) - 1):
            x0, y0 = x[i], y[i]
            x1, y1 = x[i + 1], y[i + 1]
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
        marker_colors = []
        for alpha in marker_alphas:
            color_rgba = mcolors.to_rgba(color)
            color_with_alpha = list(color_rgba)
            color_with_alpha[3] = alpha
            marker_colors.append(tuple(color_with_alpha))

        scatter = plt.scatter(x, y, color=marker_colors, s=50, zorder=3)
        # Add the original data to legend only once per topic
        if i == 0:
            legend_handles.append(scatter)

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=32)
    plt.ylabel('Topic Distribution Ratios', fontsize=32)
    # plt.title('Topic Distribution by Depth', fontsize=14)

    # Adjust legend
    #plt.legend(legend_handles, [h.get_label() for h in legend_handles], fontsize=16, loc='upper right')

    # Adjust tick font size
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_plot.pdf')

if __name__ == "__main__":
    file_name = "scs_recommended_D50_T60_common_w0_video_final_transcript_translated_topics"
    df = pd.read_csv("output/" + file_name + ".csv")

    df['topics_title'] = df['topics_title'].astype(int)
    df['topics_transcription'] = df['topics_transcription'].astype(int)

    # Filter depth 0, remove 0 relevancies
    df_depth_0 = df[df["depth"] == 0]

    df_filtered_depth_0 = filter_topics(df_depth_0, attr1="topics_title", attr2="topics_transcription", rel1=2, rel2=2)
    df_filtered_depth_0 = filter_topics(df_filtered_depth_0, attr1="topics_title", attr2="topics_transcription", rel1=2, rel2=1)
    df_filtered_depth_0 = filter_topics(df_filtered_depth_0, attr1="topics_title", attr2="topics_transcription", rel1=1, rel2=2)

    df = merge(df, df_filtered_depth_0)
    df_filtered = clean_subsequent_depth(df)

    plot_topic_distribution(df_filtered, file_name)
