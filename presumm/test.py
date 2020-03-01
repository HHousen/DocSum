import os
import sys
from collections import namedtuple
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

from modeling_bertabs import BertAbs, build_predictor
from transformers import BertTokenizer
from utils_summarization import (
    SummarizationDataset,
    process_story,
    build_mask,
    compute_token_type_ids,
    encode_for_summarization,
    fit_to_block_size,
)

def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", ". ")
        .replace("[unused2]", "")
        .strip()
    )

    return summary

def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]
    summaries = [" ".join(summary_list) for _, _, summary_list in data]

    encoded_text = [encode_for_summarization(story, summary, tokenizer) for _, story, summary in data]
    encoded_stories = torch.tensor(
        [fit_to_block_size(story, block_size, tokenizer.pad_token_id) for story, _ in encoded_text]
    )
    encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(encoded_stories),
        src=encoded_stories.to(device),
        segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        tgt_str=summaries,
    )

    return batch

def build_data_iterator(documents_dir, tokenizer, device="cuda", batch_size=4):
    dataset = SummarizationDataset(documents_dir)
    sampler = SequentialSampler(dataset)

    def collate_fn(data):
        return collate(data, tokenizer, block_size=512, device=device)

    iterator = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn,)

    return iterator

Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
model.to("cuda")
model.eval()

symbols = {
    "BOS": tokenizer.vocab["[unused0]"],
    "EOS": tokenizer.vocab["[unused1]"],
    "PAD": tokenizer.vocab["[PAD]"],
}

args = {
    "max_length": 200,
    "min_length": 50,
    "beam_size": 5,
    "alpha": 0.95,
    'block_trigram': True
}

documents_dir = "../delete"

data_iterator = build_data_iterator(documents_dir, tokenizer)
predictor = build_predictor(args, tokenizer, symbols, model)

story, summary = process_story("The Modernists were not only an sentimental but detached and aloof in other ways as well. The Post-Modernists heralded by Richard Wasson object in partic the closed worlds of Modernist art. They want a literature will reflect a looser and more realistic view of life than that i through the use of rigid artistic forms or established mythic tures. The writers singled out for attention by Wasson are mostly academics who went to school under the Modernists, and as they might, they have been unable to break their ties with their predecessors. Even the younger writers like Robert Coover and Ronald Sukenick try to try and try to break with Modernism. If a movement dies when it begins to parody itself, we can say that Nabokov's brilliant combination of fic- tion, myth, poetry, puzzle puzzle marks the beginning of the Modern Age.")
print(story, summary)
batch = collate(["test", story, summary], tokenizer, block_size=512, device="cuda")
translations = predictor.translate(batch, -1)
summaries = [format_summary(t) for t in translations]
print(summaries)

# for batch in tqdm(data_iterator):
#     translations = predictor.translate(batch, 1)
#     summaries = [format_summary(t) for t in translations]
#     print(summaries)