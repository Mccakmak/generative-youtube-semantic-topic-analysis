import pandas as pd
import transformers
import torch
from tqdm import tqdm
import re
import logging
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Suppress warnings and unnecessary print statements
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Initialize the Meta-Llama pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:2",
)

CHARACTER_LIMIT = 1000  # Limit for passage length


def find_relevance(query, passage):
    """
    Given a query and a passage, generate a relevancy score (0-3) using a shortened prompt.
    """
    combined_text = "Query: " + str(query) + "\nPassage: " + str(passage)[:CHARACTER_LIMIT]
    messages = [
        {"role": "user",
         "content": (
                 "Rate the relevance of the passage to the query on a scale from 0 to 3. "
                 "3 means the passage fully answers the query, 2 means it partially answers and less related,"
                 "1 means it is only tangentially related and not answers, and 0 means it is not relevant. "
                 "Most of the cases are not related or relevant so expect more 0 and 1 cases"
                 "Return only the integer."
                 "\n" + combined_text
         )}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=50,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=1,
    )

    output = outputs[0]["generated_text"][len(prompt):]
    # Extract an integer between 0 and 3 from the output
    match = re.search(r'\b([0-3])\b', output)
    if match:
        return int(match.group(1))
    else:
        print(f"Could not parse score from output: {output}")
        return None


def binary_map(label):
    """Map relevance scores: 0 or 1 -> 0, 2 or 3 -> 1."""
    if label in [0, 1]:
        return 0
    elif label in [2, 3]:
        return 1
    else:
        return None


if __name__ == "__main__":
    # Load the CSV file with queries, passages, and human labels
    file_name = "gpt4_evaluation_results"
    df = pd.read_csv("input/" + file_name + ".csv")

    # Use a progress bar to generate Llama ratings for each query-passage pair
    tqdm.pandas(desc="Classifying relevance using Llama")
    df["llama_rating"] = df.progress_apply(lambda row: find_relevance(row["query"], row["passage"]), axis=1)

    # Save the results with the new llama_rating column
    output_file = "output/" + file_name + "_llama_topic.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved evaluation results to {output_file}")

    # Compute binary relevance: human ratings and Llama ratings
    df["human_binary"] = df["rating"].apply(binary_map)
    df["llama_binary"] = df["llama_rating"].apply(binary_map)

    valid_df = df[df["llama_binary"].notnull()]

    accuracy = accuracy_score(valid_df["human_binary"], valid_df["llama_binary"])
    f1 = f1_score(valid_df["human_binary"], valid_df["llama_binary"])
    precision = precision_score(valid_df["human_binary"], valid_df["llama_binary"])
    recall = recall_score(valid_df["human_binary"], valid_df["llama_binary"])
    conf_matrix = confusion_matrix(valid_df["human_binary"], valid_df["llama_binary"])

    print("\nBinary Evaluation Metrics for Llama:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
