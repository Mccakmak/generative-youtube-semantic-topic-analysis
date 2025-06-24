import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use the Agg backend for Matplotlib
import matplotlib

matplotlib.use('Agg')


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


def plot_relevance_scores(df, file_name, col1, col2, col3):
    # Group by depth and calculate the mean and standard deviation relevance scores for each depth
    grouped_mean = df.groupby('depth')[[col1, col2, col3]].mean().reset_index()
    grouped_std = df.groupby('depth')[[col1, col2, col3]].std().reset_index()

    # Calculate maximum values for each column
    max_values = df[[col1, col2, col3]].max()

    # Plot the relevance scores
    plt.figure(figsize=(12, 10))

    # Plot mean relevance scores
    plt.plot(grouped_mean['depth'], grouped_mean[col1], marker='o', label='Mean Relevance Title',
             color='tab:blue', linewidth=2)

    plt.plot(grouped_mean['depth'], grouped_mean[col2], marker='o',
             label='Mean Relevance Description', color='tab:orange', linewidth=2)

    plt.plot(grouped_mean['depth'], grouped_mean[col3], marker='o',
             label='Mean Relevance Transcription', color='tab:green', linewidth=2)

    # Plot standard deviation areas with different edge styles and maximum value check
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean[col1] - grouped_std[col1]),
                     np.minimum(max_values[col1], grouped_mean[col1] + grouped_std[col1]),
                     color='tab:blue', alpha=0.2, label='Std Relevance Title')
    plt.plot(grouped_mean['depth'], np.minimum(max_values[col1], grouped_mean[col1] + grouped_std[col1]),
             color='tab:blue', linestyle='dashed', linewidth=1)
    plt.plot(grouped_mean['depth'], np.maximum(0, grouped_mean[col1] - grouped_std[col1]),
             color='tab:blue', linestyle='dashed', linewidth=1)

    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean[col2] - grouped_std[col2]),
                     np.minimum(max_values[col2], grouped_mean[col2] + grouped_std[col2]),
                     color='tab:orange', alpha=0.2, label='Std Relevance Description')
    plt.plot(grouped_mean['depth'], np.minimum(max_values[col2], grouped_mean[col2] + grouped_std[col2]),
             color='tab:orange', linestyle='dotted', linewidth=1)
    plt.plot(grouped_mean['depth'], np.maximum(0, grouped_mean[col2] - grouped_std[col2]),
             color='tab:orange', linestyle='dotted', linewidth=1)

    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean[col3] - grouped_std[col3]),
                     np.minimum(max_values[col3], grouped_mean[col3] + grouped_std[col3]),
                     color='tab:green', alpha=0.2, label='Std Relevance Transcription')
    plt.plot(grouped_mean['depth'], np.minimum(max_values[col3], grouped_mean[col3] + grouped_std[col3]),
             color='tab:green', linestyle='dashdot', linewidth=1)
    plt.plot(grouped_mean['depth'], np.maximum(0, grouped_mean[col3] - grouped_std[col3]),
             color='tab:green', linestyle='dashdot', linewidth=1)

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=26)
    plt.ylabel('Relevance Scores', fontsize=26)
    # plt.title('Relevance Scores by Depth with Standard Deviation', fontsize=14)
    plt.legend(fontsize=25)

    # Adjust tick font size
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Ensure x-axis has integer ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Show grid
    plt.grid(False)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_plot.pdf')
    # plt.show()  # Not needed for Agg backend


def plot_engagement_scores(df, file_name):
    # Apply base-10 logarithmic transformation to the engagement scores
    df.loc[:, 'total_views_log'] = np.log10(df['total_views'] + 1)  # log10(1 + x) to handle zero values
    df.loc[:, 'total_likes_log'] = np.log10(df['total_likes'] + 1)
    df.loc[:, 'total_comments_log'] = np.log10(df['total_comments'] + 1)

    # Group by depth and calculate the mean and standard deviation of the log-transformed engagement scores for each depth
    grouped_mean = df.groupby('depth')[
        ['total_views_log', 'total_likes_log', 'total_comments_log']].mean().reset_index()
    grouped_std = df.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].std().reset_index()

    # Calculate maximum values for each column
    max_values = df[['total_views_log', 'total_likes_log', 'total_comments_log']].max()

    # Plot the engagement scores
    plt.figure(figsize=(12, 10))

    # Plot mean engagement scores
    plt.plot(grouped_mean['depth'], grouped_mean['total_views_log'], marker='o', label='Mean Total Views',
             color='tab:red', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['total_likes_log'], marker='o', label='Mean Total Likes',
             color='tab:orange', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['total_comments_log'], marker='o', label='Mean Total Comments',
             color='tab:purple', linewidth=2)

    # Plot standard deviation areas with different edge styles and maximum value check
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_views_log'] - grouped_std['total_views_log']),
                     np.minimum(max_values['total_views_log'],
                                grouped_mean['total_views_log'] + grouped_std['total_views_log']),
                     color='tab:red', alpha=0.2)
    plt.plot(grouped_mean['depth'], np.minimum(max_values['total_views_log'],
                                               grouped_mean['total_views_log'] + grouped_std['total_views_log']),
             color='tab:red', linestyle='dashed', linewidth=1)
    plt.plot(grouped_mean['depth'], np.maximum(0, grouped_mean['total_views_log'] - grouped_std['total_views_log']),
             color='tab:red', linestyle='dashed', linewidth=1)

    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_likes_log'] - grouped_std['total_likes_log']),
                     np.minimum(max_values['total_likes_log'],
                                grouped_mean['total_likes_log'] + grouped_std['total_likes_log']),
                     color='tab:orange', alpha=0.2)
    plt.plot(grouped_mean['depth'], np.minimum(max_values['total_likes_log'],
                                               grouped_mean['total_likes_log'] + grouped_std['total_likes_log']),
             color='tab:orange', linestyle='dotted', linewidth=1)
    plt.plot(grouped_mean['depth'], np.maximum(0, grouped_mean['total_likes_log'] - grouped_std['total_likes_log']),
             color='tab:orange', linestyle='dotted', linewidth=1)

    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_comments_log'] - grouped_std['total_comments_log']),
                     np.minimum(max_values['total_comments_log'],
                                grouped_mean['total_comments_log'] + grouped_std['total_comments_log']),
                     color='tab:purple', alpha=0.2)
    plt.plot(grouped_mean['depth'], np.minimum(max_values['total_comments_log'],
                                               grouped_mean['total_comments_log'] + grouped_std['total_comments_log']),
             color='tab:purple', linestyle='dashdot', linewidth=1)
    plt.plot(grouped_mean['depth'],
             np.maximum(0, grouped_mean['total_comments_log'] - grouped_std['total_comments_log']),
             color='tab:purple', linestyle='dashdot', linewidth=1)

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=26)
    plt.ylabel('Log-Transformed Engagement Scores', fontsize=26)
    # plt.title('Log-Transformed Engagement Scores by Depth with Standard Deviation', fontsize=14)

    #plt.legend(fontsize=26, framealpha=0.8)

    plt.legend(fontsize=26, framealpha=0.8)

    # Adjust tick font size
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Ensure x-axis has integer ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Show grid
    plt.grid(False)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_engagement_plot.pdf')
    # plt.show()  # Not needed for Agg backend


if __name__ == "__main__":
    file_name = "taiwan_emotion_topic_relevance_final_T3"
    df = pd.read_csv("input/" + file_name + ".csv")

    # Filter depth 0, remove 0 relevancies
    # df_depth_0 = df[df["depth"] == 0]
    # df_filtered_depth_0 = filter_relevancy(df_depth_0, attr1="relevance_title", attr2="relevance_transcription", rel1=0, rel2=0)

    # df = merge(df, df_filtered_depth_0)
    # df_filtered = clean_subsequent_depth(df)

    plot_relevance_scores(df, file_name, col1="gpt_relevance_title", col2="gpt_relevance_description",
                          col3="gpt_relevance_transcription")
    plot_engagement_scores(df, file_name)
