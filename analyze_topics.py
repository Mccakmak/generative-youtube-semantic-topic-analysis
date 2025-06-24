import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    df['topics_combined'] = df[['topics_title', 'topics_transcription']].apply(lambda row: row.dropna().values.tolist(), axis=1)
    df = df.explode('topics_combined')

    # Filter out non-integer values
    df = df[df['topics_combined'].apply(lambda x: str(x).isdigit())]
    df['topics_combined'] = df['topics_combined'].astype(int)  # Ensure topics are integers

    # Group by depth and topic, then calculate the ratio for each topic
    topic_counts = df.groupby(['depth', 'topics_combined']).size().unstack(fill_value=0)
    topic_ratios = topic_counts.div(topic_counts.sum(axis=1), axis=0).reset_index()

    topic_ratios.to_csv("output/" + file_name + "_topic_probabilities.csv", index=False)

    # Plot the topic distribution
    plt.figure(figsize=(12, 10))

    # Define colors and labels for each topic (used in the graph)
    colors = {
        0: 'royalblue',   # Entertainment
        1: 'forestgreen',     # Non-Entertainment
        2: 'orange'         # Politics
    }

    # Copy of colors dictionary for the legend (you can edit this)
    legend_colors = {
        0: 'orange',   # Entertainment
        1: 'forestgreen',     # Non-Entertainment
        2: 'royalblue'         # Politics
    }

    topic_labels = {
        0: 'Entertainment',
        1: 'Non-Entertainment',
        2: 'Politics'
    }

    # Plot the ratios for each topic
    for topic in topic_ratios.columns[1:]:
        topic_int = int(topic)
        plt.plot(topic_ratios['depth'], topic_ratios[topic], marker='o',
                 label=topic_labels[topic_int], color=colors[topic_int])

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=52)
    plt.ylabel('Topic Scores', fontsize=52)
    # plt.title('Topic Distribution by Depth', fontsize=14)

    # Create custom legend handles with different colors
    legend_handles = [Line2D([0], [0], color=legend_colors[topic_int], marker='o', linestyle='-',
                             label=topic_labels[topic_int], markersize=10)
                      for topic_int in topic_ratios.columns[1:].astype(int)]

    #plt.legend(handles=legend_handles, fontsize=40)

    # Adjust tick font size
    plt.xticks(fontsize=52)
    plt.yticks(fontsize=52)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_plot.pdf')

if __name__ == "__main__":
    file_name = "taiwan_emotion_topic_relevance_final_T60"
    df = pd.read_csv("input/" + file_name + ".csv")

    df['topics_title'] = df['topics_title'].astype(int)
    df['topics_transcription'] = df['topics_transcription'].astype(int)

    # Filter depth 0, remove 0 relevancies
    #df_depth_0 = df[df["depth"] == 0]

    #df_filtered_depth_0 = filter_topics(df_depth_0, attr1="topics_title", attr2="topics_transcription", rel1=2, rel2=2)
    #df_filtered_depth_0 = filter_topics(df_filtered_depth_0, attr1="topics_title", attr2="topics_transcription", rel1=2, rel2=1)
    #df_filtered_depth_0 = filter_topics(df_filtered_depth_0, attr1="topics_title", attr2="topics_transcription", rel1=1, rel2=2)

    #df = merge(df, df_filtered_depth_0)
    #df_filtered = clean_subsequent_depth(df)

    plot_topic_distribution(df, file_name)
