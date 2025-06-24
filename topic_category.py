import pandas as pd
import transformers
import torch
from tqdm import tqdm
import logging
import warnings

# Suppress warnings and unnecessary print statements
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:7",
)

def find_topic(input_text):

    messages = [
        {"role": "user",
         "content": "You are a model that classifies a given sentence according to topic categories. Classify the given text into these topic categories: Autos & Vehicles, Comedy, Education, Entertainment, Film & Animation, Gaming, Howto & Style, Music, News & Politics, Nonprofits & Activism, People & Blogs, Pets & Animals, Science & Technology, Sports, Travel & Events" + "Input sentence:" + str(
             input_text)},
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
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    output = outputs[0]["generated_text"][len(prompt):]
    return output


if __name__ == "__main__":

    file_name = "merged_Taiwan_YT_regular_symbols_video_translated_final_transcript_unique_filtered_lama_relevance"
    df = pd.read_csv("input/" + file_name + ".csv")

    attribute = ["caption"]

    for attr in attribute:

        df_attribute = df[attr]

        tqdm.pandas(desc="Classifying topics for: " + attr)

        # For each column find relevance
        new_column = "topics_" + attr
        df[new_column] = df_attribute.progress_apply(lambda input_text: find_topic(input_text))

    df.to_csv("output/" + file_name + "general_topics.csv", index=False)





