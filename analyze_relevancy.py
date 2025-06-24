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

def normalize_column(df, column, to_range=(0, 1)):
    min_val = df[column].min()
    max_val = df[column].max()
    return to_range[0] + (df[column] - min_val) * (to_range[1] - to_range[0]) / (max_val - min_val)

def plot_relevance_scores(df, file_name):
    # Normalize relevance scores
    df.loc[:, 'relevance_title_norm'] = normalize_column(df, 'relevance_title')
    df.loc[:, 'relevance_transcription_norm'] = normalize_column(df, 'relevance_transcription')

    # Group by depth and calculate the mean and standard deviation relevance scores for each depth
    grouped_mean = df.groupby('depth')[['relevance_title_norm', 'relevance_transcription_norm']].mean().reset_index()
    grouped_std = df.groupby('depth')[['relevance_title_norm', 'relevance_transcription_norm']].std().reset_index()

    # Plot the relevance scores
    plt.figure(figsize=(12, 10))

    # Plot mean relevance scores
    plt.plot(grouped_mean['depth'], grouped_mean['relevance_title_norm'], marker='o', label='Title (Mean)',
             color='tab:blue', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['relevance_transcription_norm'], marker='o',
             label='Transcription (Mean)', color='tab:green', linewidth=2)

    # Plot standard deviation areas
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['relevance_title_norm'] - grouped_std['relevance_title_norm']),
                     grouped_mean['relevance_title_norm'] + grouped_std['relevance_title_norm'],
                     color='tab:blue', alpha=0.2, label='Title (Std)')
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['relevance_transcription_norm'] - grouped_std['relevance_transcription_norm']),
                     grouped_mean['relevance_transcription_norm'] + grouped_std['relevance_transcription_norm'],
                     color='tab:green', alpha=0.2, label='Transcription (Std)')

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=52)
    plt.ylabel('Relevance Scores', fontsize=52)
    #plt.title('Normalized Relevance Scores by Depth with Standard Deviation', fontsize=14)
    #plt.legend(fontsize=40)

    # Adjust tick font size
    plt.xticks(fontsize=52)
    plt.yticks(fontsize=52)

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
    grouped_mean = df.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].mean().reset_index()
    grouped_std = df.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].std().reset_index()
    # Plot the engagement scores
    plt.figure(figsize=(12, 10))

    # Plot mean engagement scores
    plt.plot(grouped_mean['depth'], grouped_mean['total_views_log'], marker='o', label='Mean Views',
             color='tab:red', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['total_likes_log'], marker='o', label='Mean Likes',
             color='tab:orange', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['total_comments_log'], marker='o', label='Mean Comments',
             color='tab:purple', linewidth=2)

    # Plot standard deviation areas
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_views_log'] - grouped_std['total_views_log']),
                     grouped_mean['total_views_log'] + grouped_std['total_views_log'],
                     color='tab:red', alpha=0.2, label='Std Views')
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_likes_log'] - grouped_std['total_likes_log']),
                     grouped_mean['total_likes_log'] + grouped_std['total_likes_log'],
                     color='tab:orange', alpha=0.2, label='Std Likes')
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_comments_log'] - grouped_std['total_comments_log']),
                     grouped_mean['total_comments_log'] + grouped_std['total_comments_log'],
                     color='tab:purple', alpha=0.2, label='Std Comments')

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=52)
    plt.ylabel('Engagement Scores', fontsize=52)
    #plt.title('Log-Transformed Engagement Scores by Depth with Standard Deviation', fontsize=14)

    plt.legend(fontsize=28, framealpha=1, ncol=2,  bbox_to_anchor=(0.07, 0.25), columnspacing=0.5)

    # Adjust tick font size
    plt.xticks(fontsize=52)
    plt.yticks(fontsize=52)

    # Show grid
    plt.grid(False)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_engagement_plot.pdf')
    # plt.show()  # Not needed for Agg backend

if __name__ == "__main__":
    file_name = "scs_recommended_D50_T3_common_w0_video_final_transcript_translated_emotions"
    df = pd.read_csv("output/" + file_name + ".csv")

    # Filter depth 0, remove 0 relevancies
    #df_depth_0 = df[df["depth"] == 0]
    #df_filtered_depth_0 = filter_relevancy(df_depth_0, attr1="relevance_title", attr2="relevance_transcription", rel1=0, rel2=0)

    #df = merge(df, df_filtered_depth_0)
    #df_filtered = clean_subsequent_depth(df)

    #plot_relevance_scores(df, file_name)
    plot_engagement_scores(df, file_name)
