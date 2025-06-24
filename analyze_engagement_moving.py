import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use the Agg backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

def filter_relevancy(df, attr1, attr2, rel1, rel2):
    return df.loc[~((df[attr1] == rel1) & (df[attr2] == rel2))]

def merge(df_orig, df_filtered):
    df_orig_filtered = df_orig[df_orig['depth'] != 0]
    updated_df = pd.concat([df_orig_filtered, df_filtered])
    return updated_df

def clean_subsequent_depth(df):
    root_video_ids = df[df['depth'] == 0]['root_video_id'].unique()
    filtered_df = df[df['root_video_id'].isin(root_video_ids)]
    return filtered_df

def normalize_column(df, column, to_range=(0, 1)):
    min_val = df[column].min()
    max_val = df[column].max()
    return to_range[0] + (df[column] - min_val) * (to_range[1] - to_range[0]) / (max_val - min_val)

def plot_engagement_scores(df, file_name):
    # Apply base-10 logarithmic transformation to the engagement scores
    df['total_views_log'] = np.log10(df['total_views'] + 1)  # log10(1 + x) to handle zero values
    df['total_likes_log'] = np.log10(df['total_likes'] + 1)
    df['total_comments_log'] = np.log10(df['total_comments'] + 1)

    # Group by depth and calculate the mean and standard deviation of the log-transformed engagement scores for each depth
    grouped_mean = df.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].mean().reset_index()
    grouped_std = df.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].std().reset_index()

    # Define depths where data points should be less visible
    less_visible_depths = [13, 18, 23, 28, 33, 38, 43, 48]

    # Prepare the plot
    plt.figure(figsize=(12, 10))

    # Define colors for each engagement metric
    colors = {
        'total_views_log': 'tab:red',
        'total_likes_log': 'tab:orange',
        'total_comments_log': 'tab:purple'
    }

    labels = {
        'total_views_log': 'Views',
        'total_likes_log': 'Likes',
        'total_comments_log': 'Comments'
    }

    # Prepare legend handles
    legend_handles = []

    # Plot mean engagement scores and moving averages
    for metric in ['total_views_log', 'total_likes_log', 'total_comments_log']:
        color = colors[metric]
        label = labels[metric]
        x = grouped_mean['depth'].values
        y = grouped_mean[metric].values

        # Calculate moving average
        moving_avg = pd.Series(y).rolling(window=5, center=True).mean()

        # Plot moving average with dashed line
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
                alpha = 0.4
            else:
                alpha = 1.0

            # Get RGBA color with adjusted alpha
            color_rgba = mcolors.to_rgba(color)
            color_with_alpha = list(color_rgba)
            color_with_alpha[3] = alpha
            segment_colors.append(tuple(color_with_alpha))

        # Create LineCollection for mean engagement scores
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
        # Add the original data to legend only once per metric
        legend_handles.append(scatter)

        # Plot standard deviation areas with adjusted transparency
        y_std = grouped_std[metric].values
        lower_bound = np.maximum(0, y - y_std)
        upper_bound = y + y_std

        # Adjust transparency for std areas
        std_alpha = 0.15
        std_colors = []
        for xi in x:
            alpha = std_alpha if xi not in less_visible_depths else std_alpha * 0.3
            color_rgba = mcolors.to_rgba(color)
            color_with_alpha = list(color_rgba)
            color_with_alpha[3] = alpha
            std_colors.append(tuple(color_with_alpha))

        plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=std_alpha, zorder=2)

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=32)
    plt.ylabel('Log-Transformed Engagement Scores', fontsize=32)
    # plt.title('Log-Transformed Engagement Scores by Depth with Standard Deviation', fontsize=14)

    # Create custom legend handles
    custom_legend_handles = []
    for metric in ['total_views_log', 'total_likes_log', 'total_comments_log']:
        color = colors[metric]
        label = labels[metric]
        line = Line2D([0], [0], color=color, linestyle='-', marker='o', label=label, markersize=10)
        line_avg = Line2D([0], [0], color=color, linestyle='--', label=f'{label} Moving Avg')
        custom_legend_handles.extend([line_avg, line])

    #plt.legend(handles=custom_legend_handles, fontsize=28, framealpha=1, ncol=2)

    # Adjust tick font size
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    # Show grid
    plt.grid(False)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_engagement_plot.pdf')
    # plt.show()  # Not needed for Agg backend

if __name__ == "__main__":
    file_name = "scs_recommended_D50_T60_common_w0_video_final_transcript_translated_relevance"
    df = pd.read_csv("output/" + file_name + ".csv")

    # Filter depth 0, remove 0 relevancies
    df_depth_0 = df[df["depth"] == 0]
    df_filtered_depth_0 = filter_relevancy(df_depth_0, attr1="relevance_title", attr2="relevance_transcription", rel1=0, rel2=0)

    df = merge(df, df_filtered_depth_0)
    df_filtered = clean_subsequent_depth(df)

    plot_engagement_scores(df_filtered, file_name)
