"""Zero/few-shot classification of the PT test sets with Mistral Nemo Instruct 2407.

Run through torchrun (see the Dockerfile). Environment variables:
  DATASET             - dataset id, e.g. ed_01 (required)
  NUMBER_OF_EXAMPLES  - examples per label placed in the prompt, 0 = zero-shot (default 0)
  LEAP                - index of the first test instance to predict, to resume a run (default 0)
  MODEL_DIR           - folder with the model weights and tekken.json
                        (default /data/mistral-nemo-instruct-2407)
  WORK_DIR            - root of the data; this stage reads and writes under WORK_DIR/pt (default /data)
  COLLECTION          - must be "pt" (required): there is no canonical LLM file list yet
Inputs (under WORK_DIR/pt): the LLM example CSV (stage 03) and the test CSV, both named in the
packaged pt/zero_and_few_shot_files.csv.
Outputs (under WORK_DIR/pt; ';'-separated, {LLM} = MISTRAL_NEMO_INSTRUCT_2407):
  {LLM}_{dataset}_predicted_result_{N}_shot_end_index_{i}.csv  - partial results every 3000 rows
  {LLM}_{dataset}_predicted_result_{N}_shot_final_remaining.csv
  error_{LLM}_{dataset}_predicted_result_{N}_shot.txt          - only if the run failed
"""

# %%
import datetime
import gc
from typing import Any, Union
import torch
import tiktoken
import pandas as pd
import traceback

MAX_CONTEXT_WINDOW_LENGTH = 41_000
LENGTH_LIMIT = 41_000
encoding = tiktoken.get_encoding("r50k_base")  # simplified estimation with tiktoken

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from automl_and_llm_benchmark.common.collection import current_collection, static_parts
from automl_and_llm_benchmark.common.env import get_env, get_int_env, require_env
from automl_and_llm_benchmark.common.io import read_csv
from automl_and_llm_benchmark.common.labels import load_label_map, normalize_labels
from automl_and_llm_benchmark.common.paths import output_path
from automl_and_llm_benchmark.common.resources import read_static_csv

print("Import finished")

# %%
LEAP = get_int_env("LEAP", 0)
NUMBER_OF_EXAMPLES = get_int_env("NUMBER_OF_EXAMPLES", 0)
SELECTED_DATASET = require_env("DATASET")
MODEL_DIR = get_env("MODEL_DIR", "/data/mistral-nemo-instruct-2407")
COLLECTION_NAME = current_collection()
if COLLECTION_NAME is None:
    # No canonical LLM example list exists yet: only the adjacent "pt" collection is supported.
    raise RuntimeError(
        "Stage 08 (Mistral) needs COLLECTION=pt (there is no canonical LLM file list yet); "
        "run it with `-e COLLECTION=pt` and the data under WORK_DIR/pt."
    )

SELECTED_LLM = "MISTRAL_NEMO_INSTRUCT_2407"

print(f"number of GPUs available: {torch.cuda.device_count()}")


def build_mistral() -> Union[Any]:
    return MistralTokenizer.from_file(
        f"{MODEL_DIR}/tekken.json"
    ), Transformer.from_folder(MODEL_DIR)


def call_mistral_llm(
    system_role: str,
    user_message: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_tokens: int = 20,
) -> str:
    
    # Create ChatCompletionRequest with top_p
    completion_request = ChatCompletionRequest(
        random_seed=42,
        messages=[SystemMessage(content=system_role), UserMessage(content=user_message)],
        top_p=top_p,  # Include top_p here
        temperature=temperature
    )

    # Encode the request
    all_tokens = MISTRAL_TOKENIZER.encode_chat_completion(completion_request).tokens

    # Generate text using the encoded request
    out_tokens, _ = generate(
        [all_tokens],
        MISTRAL_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        eos_id=EOS_ID,
    )

    return MISTRAL_TOKENIZER.instruct_tokenizer.tokenizer.decode(out_tokens[0])


# ed fn hs pc sd ua
# system input
system_input = {
    "ed": "You are responsible for labeling the emotions of texts.",
    "fn": "You are responsible for detecting fake news.",
    "hs": "You are responsible for detecting hate speech.",
    "pc": "You are responsible for classifying the polarity of texts.",
    "sd": "You are responsible for detecting the stance of texts.",
    "ua": "You are responsible for analyzing the utility of texts.",
}

# Cambridge dictionary
emotion_detection_labels = {
    "admiration": "the feeling of admiring someone or something",  # https://dictionary.cambridge.org/dictionary/english/admiration
    "amusement": "the feeling of being entertained or made to laugh",  # https://dictionary.cambridge.org/dictionary/english/amusement
    "anger": "a strong feeling that makes you want to hurt someone or be unpleasant because of something unfair or unkind that has happened",  # https://dictionary.cambridge.org/dictionary/english/amusement
    "annoyance": "the feeling or state of being annoyed",  # https://dictionary.cambridge.org/dictionary/english/annoyance
    "anticipation": "a feeling of excitement about something that is going to happen in the near future",  # https://dictionary.cambridge.org/dictionary/english/anticipation
    "approval": "the feeling of having a positive opinion of someone or something",  # https://dictionary.cambridge.org/dictionary/english/approval
    "caring": "a caring person is kind and gives emotional support to others",  # https://dictionary.cambridge.org/dictionary/english/caring
    "confusion": "a lack of understanding, or a state of disorder",  # https://dictionary.cambridge.org/dictionary/english/confusion
    "curiosity": "an eager wish to know or learn about something",  # https://dictionary.cambridge.org/dictionary/english/curiosity
    "desire": "to want something, especially strongly",  # https://dictionary.cambridge.org/dictionary/english/desire
    "disappointment": "something or someone that is not what you were hoping it would be",  # https://dictionary.cambridge.org/dictionary/english/disappointment
    "disapproval": "the feeling of having a negative opinion of someone or something",  # https://dictionary.cambridge.org/dictionary/english/disapproval
    "disgust": "a strong feeling of disapproval or dislike, or a feeling of becoming ill caused by something unpleasant",  # https://dictionary.cambridge.org/dictionary/english/disgust
    "embarrassment": "the feeling of being embarrassed, or something that makes you feel embarrassed",  # https://dictionary.cambridge.org/dictionary/english/embarrassment
    "excitement": "a feeling of being excited, or an exciting event",  # https://dictionary.cambridge.org/dictionary/english/excitement
    "fear": "an unpleasant emotion or thought that you have when you are frightened or worried by something dangerous, painful, or bad that is happening or might happen",  # https://dictionary.cambridge.org/dictionary/english/fear
    "gratitude": "the feeling or quality of being grateful",  # https://dictionary.cambridge.org/dictionary/english/gratitude
    "grief": "very great sadness, especially at the death of someone",  # https://dictionary.cambridge.org/dictionary/english/grief
    "joy": "great happiness",  # https://dictionary.cambridge.org/dictionary/english/joy
    "love": "the feeling of liking another adult very much and being romantically and sexually attracted to them, or strong feelings of liking a friend or person in your family",  # https://dictionary.cambridge.org/dictionary/english/love
    "nervousness": "a feeling of worry and anxiety",  # https://dictionary.cambridge.org/dictionary/english/nervousness
    "optimism": "the quality of being full of hope and emphasizing the good parts of a situation, or a belief that something good will happen",  # https://dictionary.cambridge.org/dictionary/english/optimism
    "pride": "a feeling of pleasure and satisfaction that you get because you or people connected with you have done or got something good",  # https://dictionary.cambridge.org/dictionary/english/pride
    "realization": "the fact or moment of starting to understand a situation",  # https://dictionary.cambridge.org/dictionary/english/realization
    "relief": "a feeling of happiness that something unpleasant has not happened or has ended",  # https://dictionary.cambridge.org/dictionary/english/relief
    "remorse": "a feeling of sadness and being sorry for something you have done",  # https://dictionary.cambridge.org/dictionary/english/remorse
    "sadness": "the feeling of being unhappy, especially because something bad has happened",  # https://dictionary.cambridge.org/dictionary/english/sadness
    "surprise": "an unexpected event",  # https://dictionary.cambridge.org/dictionary/english/surprise
    "neutral": "not saying or doing anything that would encourage or help any of the groups involved in an argument or war",  # https://dictionary.cambridge.org/dictionary/english/neutral
    "hope": "to want something to happen or to be true, and usually have a good reason to think that it might",  # https://dictionary.cambridge.org/dictionary/english/hope
    "not_hope": "hopeless",  # https://dictionary.cambridge.org/dictionary/english/hopeless
    "trust": "the belief that someone is good and honest and will not harm you, or that something is safe and reliable",  # https://dictionary.cambridge.org/dictionary/english/trust
    "other": "it is not about hope or a hopeless speech.",
}

fake_news_detection_labels = {  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "barely_true": "only just accurate and there's nothing significant missing",  # https://dictionary.cambridge.org/dictionary/english/barely https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "correct_attribution": "accurately saying or thinking that something is the result or work of a particular person or thing",  # https://dictionary.cambridge.org/dictionary/english/attribution
    "false": "not real, but made to look or seem real",  # https://dictionary.cambridge.org/dictionary/english/false
    "false_and_true": "accurate in some parts, but not real in others",  # https://dictionary.cambridge.org/dictionary/english/false https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "full_flop": "a complete change in position",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "half_flip": "a partial change in position",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "half_true": "partially accurate but leaves out important details or takes things out of context",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "insufficient_info": "not possible to support or refute a text segment",
    "legend": "a very old story or set of stories from ancient times, or the stories, not always true, that people tell about a famous event or person",  # https://dictionary.cambridge.org/dictionary/english/legend
    "misattributed": "to wrongly say or think that something was created, said, or written by a particular person",  # https://dictionary.cambridge.org/dictionary/english/misattribute
    "miscaptioned": "to provide the wrong piece of text under a picture instead of the text that correctly describes or explains the picture",  # adapted from https://dictionary.cambridge.org/dictionary/english/miscaption
    "misleading": "causing someone to believe something that is not true",  # https://dictionary.cambridge.org/dictionary/english/misleading
    "mostly_false": "an element of truth but ignores critical facts that would give a different impression",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "mostly_true": "accurate but needs clarification or additional information",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "no_flip": "not a significant change in position",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "no_opinion": "not saying or doing anything that would encourage or help true or false claims",  # adapted from neutral definition https://dictionary.cambridge.org/dictionary/english/neutral
    "out_of_scope": "unrelated or barely related to true or false claims",
    "outdated": "no longer useful or accurate",  # adapted from https://dictionary.cambridge.org/dictionary/english/outdated
    "pants_on_fire": "not accurate and makes a ridiculous claim",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "partly_false": "part of it is not real, but made to look or seem real",  # https://dictionary.cambridge.org/dictionary/english/false
    "refutation": "the act of saying or proving that a person, statement, opinion, etc. is wrong or false",  # https://dictionary.cambridge.org/dictionary/english/refutation
    "scam": "a dishonest plan for making money or getting an advantage, especially one that involves tricking people",  # https://dictionary.cambridge.org/dictionary/english/scam
    "support": "to help to show something to be true",  # https://dictionary.cambridge.org/dictionary/english/support
    "true": "accurate and there's nothing significant missing",  # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
    "unproven": "not having been shown to be true",  # https://dictionary.cambridge.org/dictionary/english/unproven
}

hate_speech_detection_labels = {
    "appropriative": "taking something for your own use, usually without permission",  # https://dictionary.cambridge.org/dictionary/english/appropriation
    "derogatory": "showing strong disapproval and not showing respect",  # https://dictionary.cambridge.org/dictionary/english/derogatory
    "hate": "an extremely strong dislike",  # https://dictionary.cambridge.org/dictionary/english/hate
    "homonym": "a word that is spelled the same as another word but that does not have the same meaning",  # https://dictionary.cambridge.org/dictionary/english/homonym
    "neither": "not an extremely strong dislike and is not causing offence",  # https://dictionary.cambridge.org/dictionary/english/hate https://dictionary.cambridge.org/dictionary/english/offensive
    "neutral": "not saying or doing anything that would encourage or help any of the groups involved in an argument or war",  # https://dictionary.cambridge.org/dictionary/english/neutral
    "non_derogatory_non_appropriative": "not taking something for your own use, usually without permission and is not showing strong disapproval and not showing respect",  # https://dictionary.cambridge.org/dictionary/english/appropriation https://dictionary.cambridge.org/dictionary/english/derogatory
    "not_hate": "not an extremely strong dislike",  # https://dictionary.cambridge.org/dictionary/english/hate
    "not_toxic": "not unpleasant or unacceptable",  # https://dictionary.cambridge.org/dictionary/english/toxic
    "noise": "not related to hate or offensive speech",
    "offensive": "causing offence",  # https://dictionary.cambridge.org/dictionary/english/offensive
    "sexism": "the belief that the members of one sex are less intelligent, able, skilful, etc. than the members of the other sex, especially that women are less able than men",  # https://dictionary.cambridge.org/dictionary/english/sexism
    "spam": "unwanted email, usually advertisements",  # https://dictionary.cambridge.org/dictionary/english/spam
    "toxic": "very unpleasant or unacceptable",  # https://dictionary.cambridge.org/dictionary/english/toxic
}

stance_detection_labels = {
    "not_hyperpartisan": "not an extremely partisan and extremely biased in favor of a political party",  # https://en.wiktionary.org/wiki/hyperpartisan
    "hyperpartisan": "extremely partisan and extremely biased in favor of a political party",  # https://en.wiktionary.org/wiki/hyperpartisan
    "right": "the political parties or people that have traditional opinions, and that believe in low taxes, property, and industry being privately owned, and less help for the poor",  # https://dictionary.cambridge.org/dictionary/english/right
    "right_center": "a political belief that contains some conservative ideas but is not extreme",  # https://dictionary.cambridge.org/dictionary/english/right-of-centre
    "least": "not saying or doing anything that would encourage or help any of the political groups",  # adapted from neutral definition https://dictionary.cambridge.org/dictionary/english/neutral
    "left-center": "a political belief that contains some socialist ideas but is not extreme",  # https://dictionary.cambridge.org/dictionary/english/left-of-centre
    "left": "the political groups that believe wealth and power should be shared between all parts of society",  # https://dictionary.cambridge.org/dictionary/english/left
    "neutral": "not saying or doing anything that would encourage or help any of the groups",  # adapted from neutral definition https://dictionary.cambridge.org/dictionary/english/neutral
    "against": "opposed to someone or something",  # adapted from https://dictionary.cambridge.org/dictionary/english/against
    "favorable": "when support or approve of someone or something",  # adapted from https://dictionary.cambridge.org/dictionary/english/favourable
}

utility_analysis_labels = {
    "1": "extremely low",
    "2": "very low",
    "3": "low",
    "4": "slightly below average",
    "5": "average",
    "6": "slightly above average",
    "7": "good",
    "8": "very good",
    "9": "excellent",
    "10": "optimal",
}

polarity_classification_labels = {
    "negative": "bad or harmful",  # https://dictionary.cambridge.org/dictionary/english/negative
    "positive": "full of hope and confidence",  # https://dictionary.cambridge.org/dictionary/english/positive
    "neutral": "not saying or doing anything that would encourage or help any of the groups",  # adapted from neutral definition https://dictionary.cambridge.org/dictionary/english/neutral
    "mixed": "positive and negative opinions",
    'very_low': ' to a great degree or extremely of bad quality, not as good as it should be', # https://dictionary.cambridge.org/dictionary/english/very https://dictionary.cambridge.org/dictionary/english/low
    'low': 'bad quality, not as good as it should be', #  https://dictionary.cambridge.org/dictionary/english/low
    'slightly_below_average': 'a little under a standard or level that is considered to be typical or usual', # https://dictionary.cambridge.org/dictionary/english/slightly https://dictionary.cambridge.org/dictionary/english/below 
    'average': 'a standard or level that is considered to be typical or usual', # https://dictionary.cambridge.org/dictionary/english/average
    'very_good': 'to a great degree or extremely satisfactory, enjoyable, pleasant, or interesting', # https://dictionary.cambridge.org/dictionary/english/very https://dictionary.cambridge.org/dictionary/english/good
    'optimal': 'the best or most effective possible in a particular situation' # https://dictionary.cambridge.org/dictionary/english/optimal
}

selected_dictionary_with_labels = {
    "ed": emotion_detection_labels,
    "fn": fake_news_detection_labels,
    "hs": hate_speech_detection_labels,
    "pc": polarity_classification_labels,
    "sd": stance_detection_labels,
    "ua": utility_analysis_labels,
}


# %%
MISTRAL_TOKENIZER, MISTRAL_MODEL = build_mistral()
EOS_ID = MISTRAL_TOKENIZER.instruct_tokenizer.tokenizer.eos_id
print("Mistral built!")

try:

    df_files = read_static_csv(*static_parts(COLLECTION_NAME, "zero_and_few_shot_files.csv"))
    filtered_files_df = df_files[df_files["name_dataset"] == SELECTED_DATASET]
    collect_results_list = []

    df = read_csv(filtered_files_df["name_file_llm"].iloc[0])

    df_predict = read_csv(filtered_files_df["name_file_predict"].iloc[0])
    df_predict = df_predict.dropna()

    df_predict["label"] = normalize_labels(df_predict["label"])
    dict_replacement = load_label_map(
        filtered_files_df["name_dataset"].iloc[0], collection=COLLECTION_NAME
    )
    df_predict["label"] = df_predict["label"].replace(dict_replacement)

    collected_labels = df["label"].unique().tolist()
    collected_labels = [str(item).lower() for item in collected_labels]
    try:
        collected_labels = [int(item) for item in collected_labels]
        collected_labels.sort()
        collected_labels = [str(item) for item in collected_labels]
    except Exception:
        collected_labels.sort()

    collected_labels_str = ", ".join(collected_labels)
    description_task = f"It is necessary to classify a text segment using one of the {len(collected_labels)} labels: {collected_labels_str}.\nThe following lines mention the labels and define them.\n"

    task = filtered_files_df["name_dataset"].iloc[0].split("_")[0]
    CONTEXT_SENTENCE = system_input[task]
    task_dictionary = selected_dictionary_with_labels[task]

    base_prompt = ""
    base_prompt = base_prompt + f"{CONTEXT_SENTENCE}\n{description_task}"

    TOTAL_TOKENS_CURRENT_PROMPT = len(encoding.encode(base_prompt))

    label_definition_example_dict = {}
    label_definition_example_dict = {
        item: [
            f"\nLabel: {item}",
            f"Definition: {task_dictionary[item]} \nExamples:",
        ]
        for item in collected_labels
    }

    for element in label_definition_example_dict.values():
        TOTAL_TOKENS_CURRENT_PROMPT += len(encoding.encode("\n".join(element)))

    if task == "ua":
        df["label"] = df["label"].astype(str)
    for count_instance in range(NUMBER_OF_EXAMPLES):
        if (
            TOTAL_TOKENS_CURRENT_PROMPT < MAX_CONTEXT_WINDOW_LENGTH
        ):  # defined max context window length
            for label in collected_labels:

                df_filtered = df[df["label"] == label]
                examples_list = list(df_filtered["text"])
                try:
                    instance_example = examples_list[count_instance]
                    number_tokens_instance = len(encoding.encode(instance_example))
                except IndexError:
                    continue
                if (
                    TOTAL_TOKENS_CURRENT_PROMPT + number_tokens_instance
                ) > MAX_CONTEXT_WINDOW_LENGTH:
                    break
                label_definition_example_dict[label].append(instance_example)
                TOTAL_TOKENS_CURRENT_PROMPT += number_tokens_instance

        else:
            break

    for selected_label in collected_labels:
        base_prompt = (
            base_prompt
            + "\n"
            + "\n".join(label_definition_example_dict[selected_label])
            + "\n"
        )

    base_prompt = base_prompt + "\nLabel the following text: '"

    instances_to_predict_list = list(df_predict["text"])

    for counter_pred, instance in enumerate(instances_to_predict_list[LEAP:], start=LEAP):
        if (counter_pred % 3000 == 0) and (counter_pred > 0):
            print(f"{counter_pred=} {datetime.datetime.now()}")
            del MISTRAL_MODEL
            del MISTRAL_TOKENIZER
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            MISTRAL_TOKENIZER, MISTRAL_MODEL = build_mistral()
            EOS_ID = MISTRAL_TOKENIZER.instruct_tokenizer.tokenizer.eos_id

        final_prompt = base_prompt + f"{instance}'\nReturn only the selected label."

        num_tokens = len(encoding.encode(final_prompt))
        if num_tokens > LENGTH_LIMIT:
            collect_results_list.append(
                [
                    filtered_files_df["name_dataset"].iloc[0],
                    df_predict["label"].iloc[counter_pred],
                    "",
                    counter_pred,
                    SELECTED_LLM,
                    NUMBER_OF_EXAMPLES,
                ]
            )
            print(f"row number: {counter_pred} prompt is too lengthy: {num_tokens}")
        else:
            output_llm_str = call_mistral_llm(
                system_role=CONTEXT_SENTENCE, user_message=final_prompt
            )
            collect_results_list.append(
                [
                    filtered_files_df["name_dataset"].iloc[0],
                    df_predict["label"].iloc[counter_pred],
                    output_llm_str,
                    counter_pred,
                    SELECTED_LLM,
                    NUMBER_OF_EXAMPLES,
                ]
            )
        if (counter_pred % 3000 == 0) and (counter_pred > 0):
            predicted_result_df = pd.DataFrame(collect_results_list)
            predicted_result_df.columns = [
                "name_dataset",
                "hold_out_y_true",
                "hold_out_y_pred",
                "index_number",
                "selected_llm",
                "number_of_examples",
            ]
            predicted_result_df.to_csv(
                output_path(
                    f"{SELECTED_LLM}_{filtered_files_df['name_dataset'].iloc[0]}_predicted_result_{NUMBER_OF_EXAMPLES}_shot_end_index_{counter_pred}.csv"
                ),
                sep=";",
                index=False,
            )
            del collect_results_list
            del predicted_result_df
            print(f"{counter_pred}: {datetime.datetime.now()}")
            torch.cuda.empty_cache()  # Free unused memory
            torch.cuda.ipc_collect()  # Helps reclaim fragmented memory
            gc.collect()
            collect_results_list = []

    predicted_result_df = pd.DataFrame(collect_results_list)
    predicted_result_df.columns = [
        "name_dataset",
        "hold_out_y_true",
        "hold_out_y_pred",
        "index_number",
        "selected_llm",
        "number_of_examples",
    ]
    predicted_result_df.to_csv(
        output_path(
            f"{SELECTED_LLM}_{filtered_files_df['name_dataset'].iloc[0]}_predicted_result_{NUMBER_OF_EXAMPLES}_shot_final_remaining.csv"
        ),
        sep=";",
        index=False,
    )

except Exception as e:
    with open(
        output_path(
            f"error_{SELECTED_LLM}_{filtered_files_df['name_dataset'].iloc[0]}_predicted_result_{NUMBER_OF_EXAMPLES}_shot.txt"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"Error: {e}\n{traceback.format_exc()}")
