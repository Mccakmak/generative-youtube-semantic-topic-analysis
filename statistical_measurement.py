import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import euclidean

# Load the data
file_path = 'output/scs_recommended_D50_T60_common_w0_video_final_transcript_translated_emotions_emotions_probabilities.csv'
data = pd.read_csv(file_path)

# Optional
#data = data.drop(columns=['1', '4'])

# Define a function to calculate the Jensen-Shannon divergence between two probability distributions
def calculate_jsd(p, q):
    return jensenshannon(p, q, base=2)

# Define a function to calculate the Hellinger distance between two probability distributions
def calculate_hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

# Extract the probability distributions for depth 0
depth_0_probs = data[data['depth'] == 0].iloc[0, 1:].values

# Initialize lists to store the JSD and Hellinger scores
jsd_scores = []
hellinger_scores = []

# Calculate the JSD and Hellinger distance for each depth with depth 0
for depth in data['depth'].unique():
    if depth == 0:
        continue
    depth_probs = data[data['depth'] == depth].iloc[0, 1:].values
    jsd_score = calculate_jsd(depth_0_probs, depth_probs)
    hellinger_score = calculate_hellinger(depth_0_probs, depth_probs)
    jsd_scores.append(jsd_score)
    hellinger_scores.append(hellinger_score)

# Calculate the mean JSD and Hellinger scores
mean_jsd_score = np.mean(jsd_scores)
mean_hellinger_score = np.mean(hellinger_scores)

# Display the mean JSD and Hellinger scores
print(f"The mean JSD score is: {mean_jsd_score}")
print(f"The mean Hellinger score is: {mean_hellinger_score}")
