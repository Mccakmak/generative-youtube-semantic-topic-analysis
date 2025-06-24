import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
symbols_df = pd.read_csv('output/merged_Taiwan_YT_regular_symbols_video_translated_final_transcript_unique_filtered_lama_relevance_topic.csv')
no_symbols_df = pd.read_csv('output/merged_Taiwan_YT_regular_no_symbols_video_translated_final_transcript_unique_filtered_lama_relevance_topic.csv')
file_name = 'Taiwan'
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
    plt.plot(grouped_mean['depth'], grouped_mean['total_views_log'], marker='o', label='Mean Total Views',
             color='tab:red', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['total_likes_log'], marker='o', label='Mean Total Likes',
             color='tab:orange', linewidth=2)
    plt.plot(grouped_mean['depth'], grouped_mean['total_comments_log'], marker='o', label='Mean Total Comments',
             color='tab:purple', linewidth=2)

    # Plot standard deviation areas
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_views_log'] - grouped_std['total_views_log']),
                     grouped_mean['total_views_log'] + grouped_std['total_views_log'],
                     color='tab:red', alpha=0.2, label='Std Total Views')
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_likes_log'] - grouped_std['total_likes_log']),
                     grouped_mean['total_likes_log'] + grouped_std['total_likes_log'],
                     color='tab:orange', alpha=0.2, label='Std Total Likes')
    plt.fill_between(grouped_mean['depth'],
                     np.maximum(0, grouped_mean['total_comments_log'] - grouped_std['total_comments_log']),
                     grouped_mean['total_comments_log'] + grouped_std['total_comments_log'],
                     color='tab:purple', alpha=0.2, label='Std Total Comments')

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=26)
    plt.ylabel('Log(10)-Transformed Engagements', fontsize=26)
    #plt.title('Log-Transformed Engagement Scores by Depth with Standard Deviation', fontsize=14)

    #plt.legend(fontsize=22, framealpha=0.8)

    # Adjust tick font size
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Show grid
    plt.grid(False)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_engagement_plot.pdf')
    # plt.show()  # Not needed for Agg backend

def plot_engagement_scores_combined_bar(df1, df2, file_name1, file_name2):
    # Apply base-10 logarithmic transformation to the engagement scores
    for df in [df1, df2]:
        df.loc[:, 'total_views_log'] = np.log10(df['total_views'] + 1)  # log10(1 + x) to handle zero values
        df.loc[:, 'total_likes_log'] = np.log10(df['total_likes'] + 1)
        df.loc[:, 'total_comments_log'] = np.log10(df['total_comments'] + 1)

    # Group by depth and calculate the mean for each dataframe
    grouped_mean1 = df1.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].mean().reset_index()
    grouped_mean2 = df2.groupby('depth')[['total_views_log', 'total_likes_log', 'total_comments_log']].mean().reset_index()

    # Set up bar plot
    fig, ax = plt.subplots(figsize=(12, 10))
    bar_width = 0.1  # Width of each bar
    spacing = 0.05  # Additional spacing between groups
    depths = grouped_mean1['depth']
    x = np.arange(len(depths))  # Original positions for the depth groups

    # Calculate new positions for the bars to align with depth labels
    x_new = x - (2 * bar_width + spacing) / 2  # Center of the grouped bars

    # Plot the bar charts for each metric

    ax.bar(x_new - bar_width - spacing, grouped_mean1['total_views_log'], bar_width, label=f'Symbol Views', color='lightblue')
    ax.bar(x_new - spacing, grouped_mean2['total_views_log'], bar_width, label=f'No Symbol Views', color='lightblue', hatch='//')

    ax.bar(x_new + bar_width + spacing, grouped_mean1['total_likes_log'], bar_width, label=f'Symbol Likes', color='green')
    ax.bar(x_new + 2 * bar_width + spacing, grouped_mean2['total_likes_log'], bar_width, label=f'No Symbol Likes', color='green', hatch='//')

    ax.bar(x_new + 3 * bar_width + 2 * spacing, grouped_mean1['total_comments_log'], bar_width, label=f'Symbol Comments', color='orange')
    ax.bar(x_new + 4 * bar_width + 2 * spacing, grouped_mean2['total_comments_log'], bar_width, label=f'No Symbol Comments', color='orange', hatch='//')

    # Add labels and title with font size adjustments
    ax.set_xlabel('Depth', fontsize=26)
    ax.set_ylabel('Log(10)-Transformed Engagements', fontsize=26)
    ax.set_xticks(x)  # Original depth positions
    ax.set_xticklabels(depths)  # Depth labels

    # Adjust legend to the right side
    plt.legend(fontsize=16, framealpha=1, loc='center left', bbox_to_anchor=(0.82, 0.8))

    # Adjust tick font size
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Show grid
    plt.grid(False)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name1 + '_vs_' + file_name2 + '_engagement_combined_bar_plot.pdf')



# Call the functions to plot
plot_engagement_scores(symbols_df, file_name + "_symbols")
plot_engagement_scores(no_symbols_df, file_name + "_no_symbols")
plot_engagement_scores_combined_bar(symbols_df, no_symbols_df, file_name + "_symbols", file_name + "_no_symbols")

