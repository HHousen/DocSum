# TODO: Get module/package imports working correctly

import os
import sys
from collections import namedtuple
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

from .modeling_bertabs import BertAbs, build_predictor
from transformers import BertTokenizer
from .utils_summarization import (
    SummarizationDataset,
    process_story,
    build_mask,
    compute_token_type_ids,
    encode_for_summarization,
    fit_to_block_size,
)


class PreSummSummarizer():
    def __init__(self, batch_size=4, device=None):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
        model.to(device)
        model.eval()

        symbols = {
            "BOS": tokenizer.vocab["[unused0]"],
            "EOS": tokenizer.vocab["[unused1]"],
            "PAD": tokenizer.vocab["[PAD]"],
        }

        self.Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])

        self.tokenizer = tokenizer
        self.model = model
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device

    def collate(self, data, tokenizer, block_size, device):
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

        batch = self.Batch(
            document_names=names,
            batch_size=len(encoded_stories),
            src=encoded_stories.to(device),
            segs=encoder_token_type_ids.to(device),
            mask_src=encoder_mask.to(device),
            tgt_str=summaries,
        )

        return batch

    @staticmethod
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
    
    @staticmethod
    def save_summaries(summaries, path, original_document_name):
        """ Write the summaries in files that are prefixed by the original
        files' name with the `_summary` appended.

        Attributes:
            original_document_names: List[string]
                Name of the document that was summarized.
            path: string
                Path were the summaries will be written
            summaries: List[string]
                The summaries that we produced.
        """
        for summary, document_name in zip(summaries, original_document_name):
            # Prepare the summary file's name
            if "." in document_name:
                bare_document_name = ".".join(document_name.split(".")[:-1])
                extension = document_name.split(".")[-1]
                name = bare_document_name + "_summary." + extension
            else:
                name = document_name + "_summary"

            file_path = os.path.join(path, name)
            with open(file_path, "w") as output:
                output.write(summary)

    def summarize_folder(self, documents_dir, summaries_output_dir, max_len_a=None, max_len_b=200,
                         min_len=50, beam_size=5, alpha=0.95, block_trigram=True):
        args = {
            "max_length": max_len_b,
            "min_length": min_len,
            "beam_size": beam_size,
            "alpha": alpha,
            'block_trigram': block_trigram
        }

        predictor = build_predictor(args, self.tokenizer, self.symbols, self.model)

        data_iterator = self.build_data_iterator(documents_dir)
        for batch in tqdm(data_iterator):
            translations = predictor.translate(batch, -1)
            summaries = [self.format_summary(t) for t in translations]
            self.save_summaries(summaries, summaries_output_dir, batch.document_names)
    
    def summarize_string(self, input_string, max_len_a=None, max_len_b=200, min_len=50,
                         beam_size=5, alpha=0.95, block_trigram=True):
        args = {
            "max_length": max_len_b,
            "min_length": min_len,
            "beam_size": beam_size,
            "alpha": alpha,
            'block_trigram': block_trigram
        }

        predictor = build_predictor(args, self.tokenizer, self.symbols, self.model)        

        story, summary = process_story(input_string)
        batch = self.collate([["useless_name", story, summary]], self.tokenizer, block_size=512, device=self.device)
        translations = predictor.translate(batch, -1)
        summaries = [self.format_summary(t) for t in translations]
        return summaries[0]
    
    def build_data_iterator(self, documents_dir):
        dataset = SummarizationDataset(documents_dir)
        sampler = SequentialSampler(dataset)

        def collate_fn(data):
            return self.collate(data, self.tokenizer, block_size=512, device=self.device)

        iterator = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=collate_fn,)

        return iterator