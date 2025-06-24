import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
symbols_df = pd.read_csv('output/merged_Taiwan_YT_regular_symbols_video_translated_final_transcript_unique_filtered_lama_relevance.csv')
no_symbols_df = pd.read_csv('output/merged_Taiwan_YT_regular_no_symbols_video_translated_final_transcript_unique_filtered_lama_relevance.csv')
model = "lama"
file = "Taiwan_relevance"

# Combine the datasets and add a column to distinguish between them
no_symbols_df['dataset'] = 'No Symbols'
symbols_df['dataset'] = 'With Symbols'

combined_df = pd.concat([no_symbols_df, symbols_df])

# Function to create bar plots
def plot_relevance_scores(model, attribute, title):
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='depth',
        y=f'{model}_relevance_{attribute}',
        hue='dataset',
        data=combined_df,
        palette='bright'
    )
    plt.xlabel('Depth', fontsize = 36)
    plt.ylabel('Relevancy Scores', fontsize = 36)
    #plt.title(title)
    plt.legend(fontsize = 36)
    #plt.show()
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.tight_layout()
    plt.savefig(f'results/{file}_{model}_{attribute}_relevance_box_plot.pdf')
    plt.close()





# Plot for each attribute
plot_relevance_scores(model, 'title', 'Relevance Scores for GPT Title by Depth')
plot_relevance_scores(model, 'description', 'Relevance Scores for GPT Description by Depth')
plot_relevance_scores(model, 'transcription', 'Relevance Scores for GPT Transcription by Depth')

