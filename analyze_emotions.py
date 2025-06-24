import pandas as pd
import matplotlib.pyplot as plt

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
    df['emotions_combined'] = df[['emotion_title', 'emotion_transcription']].apply(
        lambda row: row.dropna().values.tolist(), axis=1)
    df = df.explode('emotions_combined')

    # Filter out non-integer values
    df = df[df['emotions_combined'].apply(lambda x: str(x).isdigit())]
    df['emotions_combined'] = df['emotions_combined'].astype(int)  # Ensure emotions are integers

    # Group by depth and emotions, then calculate the ratio for each emotion
    emotions_counts = df.groupby(['depth', 'emotions_combined']).size().unstack(fill_value=0)
    emotions_ratios = emotions_counts.div(emotions_counts.sum(axis=1), axis=0).reset_index()

    emotions_ratios.to_csv("output/" + file_name + "_emotions_probabilities.csv", index=False)

    # Plot the emotions distribution
    plt.figure(figsize=(12, 10))

    # Define colors for each emotion (used in the graph)
    colors = {
        0: 'orange',       # Joy/Happiness
        1: 'royalblue',  # Sadness
        2: 'red',          # Anger
        3: 'forestgreen',    # Neutral
        4: 'indigo'        # Fear
    }

    # Copy of colors dictionary for the legend (you can edit this)
    legend_colors = {
        0: 'orange',       # Joy/Happiness
        1: 'forestgreen',  # Sadness
        2: 'red',          # Anger
        3: 'royalblue',    # Neutral
        4: 'indigo'        # Fear
    }

    # Emotion labels
    emotion_labels = {
        0: 'Joy/Happiness',
        1: 'Neutral',
        2: 'Anger',
        3: 'Sadness',
        4: 'Fear'
    }

    # Plot the ratios for each emotion
    for emotion in emotions_ratios.columns[1:]:
        emotion_int = int(emotion)
        plt.plot(emotions_ratios['depth'], emotions_ratios[emotion], marker='o',
                 label=emotion_labels[emotion_int], color=colors[emotion_int])

    # Add labels and title with font size adjustments
    plt.xlabel('Depth', fontsize=52)
    plt.ylabel('Emotion Scores', fontsize=52)
    # plt.title('Emotion Distribution by Depth', fontsize=14)

    # Create custom legend handles with different colors
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=legend_colors[emotion_int], marker='o', linestyle='-',
                             label=emotion_labels[emotion_int], markersize=7)
                      for emotion_int in emotions_ratios.columns[1:].astype(int)]

    plt.legend(handles=legend_handles, fontsize=38, loc='upper right', bbox_to_anchor=(1, 0.88))

    # Adjust tick font size
    plt.xticks(fontsize=52)
    plt.yticks(fontsize=52)

    # Display the plot with tight layout and save as PDF
    plt.tight_layout()
    plt.savefig('results/' + file_name + '_plot.pdf')

if __name__ == "__main__":
    file_name = "scs_recommended_D50_T3_common_w0_video_final_transcript_translated_emotions"
    df = pd.read_csv("output/" + file_name + ".csv")

    df['emotion_title'] = df['emotion_title'].astype(int)
    df['emotion_transcription'] = df['emotion_transcription'].astype(int)

    # Filter depth 0, remove 0 relevancies
    #df_depth_0 = df[df["depth"] == 0]

    #df_filtered_depth_0 = filter_emotions(df_depth_0, attr1="emotion_title", attr2="emotion_transcription", rel1=0, rel2=0)

    #df = merge(df, df_filtered_depth_0)
    #df_filtered = clean_subsequent_depth(df)

    plot_emotion_distribution(df, file_name)
