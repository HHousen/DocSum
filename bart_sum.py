import torch

class BartSumSummarizer():
    def __init__(self):
        bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
        bart.cuda()
        bart.eval()
        bart.half()
        
        self.bart = bart

    def summarize_string(self, source_line, min_len=55, max_len_a=0, max_len_b=140):
        """Summarize a single document"""
        source_line = [source_line]
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
