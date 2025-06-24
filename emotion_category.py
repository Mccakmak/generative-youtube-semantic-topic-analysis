import pandas as pd
import transformers
import torch
from tqdm import tqdm
import logging
import warnings
import gc

# Suppress warnings and unnecessary print statements
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:6",
)

def truncate_input_text(input_text, max_length=512):
    # Check if input_text is NaN or None
    if pd.isnull(input_text):
        input_text = ''
    else:
        input_text = str(input_text)

    input_ids = pipeline.tokenizer.encode(
        input_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )
    truncated_text = pipeline.tokenizer.decode(input_ids)
    return truncated_text

def find_topic(input_text):
    # Ensure input_text is valid
    if pd.isnull(input_text):
        input_text = ''
    else:
        input_text = str(input_text)

    # Apply truncation
    input_text = truncate_input_text(input_text)

    messages = [
        {
            "role": "user",
            "content": (
                "Classify the sentence's emotion and output only its label number: for happiness/joy print 0, sadness print 1, anger print 2, for neutral print 0, 4=fear print 0, 5=disgust print 0, 6=surprise print 0. " + "Input sentence: " + input_text
            ),
        }
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
        max_new_tokens=50,  # Reduced from 256
        eos_token_id=terminators,
        do_sample=False,    # Deterministic output
        temperature=0.6,
        top_p=0.9,
    )

    output = outputs[0]["generated_text"][len(prompt):]

    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

    return output

if __name__ == "__main__":

    file_name = "merged_Taiwan_YT_regular_symbols_video_translated_final_transcript_unique_filtered_lama_relevance"
    df = pd.read_csv("input/" + file_name + ".csv")

    attribute = ["title", "description", "transcription"]

    for attr in attribute:

        df_attribute = df[attr]

        tqdm.pandas(desc="Classifying emotions for: " + attr)

        new_column = "emotion_" + attr
        df[new_column] = df_attribute.progress_apply(lambda input_text: find_topic(input_text))

    df.to_csv("output/" + file_name + "_emotions.csv")
