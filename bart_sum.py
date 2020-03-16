import os
from pathlib import Path
import appdirs
import gdown
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class BartSumSummarizer():
    def __init__(self, device=None, checkpoint=None, state_dict_key='model', hg_transformers=True):
        if not hg_transformers and checkpoint:
            raise Exception("hg_transformers must be set to True in order to load from checkpoint")

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if checkpoint != None and "semsim" in checkpoint:
            cache_dir = appdirs.user_cache_dir("DocSum", "HHousen")
            output_file_path = os.path.join(cache_dir, "bart_semsim.pt")
            if not os.path.isfile(output_file_path):
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                gdown.download("https://drive.google.com/uc?id=1CNgK6ZkaqUD239h_6GkLmfUOGgryc2v9", output_file_path)
            checkpoint = output_file_path

        if checkpoint:
            loaded_checkpoint = torch.load(checkpoint)
            model_state_dict = loaded_checkpoint[state_dict_key]

            bart = BartForConditionalGeneration.from_pretrained("bart-large-cnn", state_dict=model_state_dict)
            tokenizer = BartTokenizer.from_pretrained("bart-large-cnn", state_dict=model_state_dict)
            self.tokenizer = tokenizer
        else:
            if hg_transformers:
                bart = BartForConditionalGeneration.from_pretrained("bart-large-cnn")
                tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")
                self.tokenizer = tokenizer
            else:
                bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
                bart.to(device)
                bart.eval()
                bart.half()
        
        self.hg_transformers = hg_transformers
        self.bart = bart

    def summarize_string(self, source_line, min_len=55, max_len_a=0, max_len_b=140):
        """Summarize a single document"""
        source_line = [source_line]

        if self.hg_transformers:
            inputs = self.tokenizer.batch_encode_plus(source_line, max_length=1024, return_tensors='pt')
            # Generate Summary
            summary_ids = self.bart.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=4, max_length=max_len_b)

            return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        else:
            with torch.no_grad():
                # beam = beam size
                # lenpen = length penalty: <1.0 favors shorter, >1.0 favors longer sentences
                # max_len_a & max_len_b = generate sequences of maximum length ax + b, where x is the source length
                # min_len = minimum generation length
                # no_repeat_ngram_size = ngram blocking such that this size ngram cannot be repeated in the generation
                # https://fairseq.readthedocs.io/en/latest/command_line_tools.html
                # print("max_len_b " + str(max_len_b) + "      min_len " + str(min_len))
                hypotheses = self.bart.sample(source_line, beam=4, lenpen=2.0, max_len_a=max_len_a, max_len_b=max_len_b, min_len=min_len, no_repeat_ngram_size=3)
            return hypotheses[0]
