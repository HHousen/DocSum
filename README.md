![DocSum Logo](docsum.png)
# DocSum
> A tool to automatically summarize documents (or plain text) using either the BART or PreSumm Machine Learning Model.

**BART** ([BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)) is the state-of-the-art in text summarization as of 02/02/2020. It is a "sequence-to-sequence model trained with denoising as pretraining objective" ([Documentation & Examples](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.md)).

**PreSumm** ([Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf)) applies BERT (Bidirectional Encoder Representations from Transformers) to text summarization by using "a novel document-level encoder based on BERT which is able to express the semantics of a document and obtain representations for its sentences." BERT represented "the latest incarnation of pretrained language models which have recently advanced a wide range of natural language processing tasks" at the time of writing ([Documentation & Examples](https://github.com/nlpyang/PreSumm)).

## Tasks

1. Convert a PDF to XML and then interpret that XML file using the `font` property of each `text` element using [main.py](main.py). Utilizes the [xml.etree.elementtree](https://docs.python.org/3/library/xml.etree.elementtree.html) python library.
2. Summarize raw text input using [cmd_summarizer.py](cmd_summarizer.py)
3. Summarize multiple text files using [presumm/run_summarization.py](presumm/run_summarization.py)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites
* [Python](https://www.python.org/)
* [Git](https://git-scm.com/)
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

### Installation

```bash
sudo apt install poppler-utils
git clone https://github.com/HHousen/docsum.git
cd docsum
conda env create --file environment.yml
conda activate docsum
```

### To convert PDF to XML

```
pdftohtml input.pdf -i -s -c -xml output.xml
```

## Project Structure
```bash
DocSum
├── bart_sum.py
├── cmd_summarizer.py
├── docsum.png
├── environment.yml
├── LICENSE
├── main.py
├── presumm
│   ├── configuration_bertabs.py
│   ├── __init__.py
│   ├── modeling_bertabs.py
│   ├── presumm.py
│   ├── run_summarization.py
│   └── utils_summarization.py
├── README.md
└── xml_processor.py
```

## Usage
Output of `python main.py --help`:
```
usage: main.py [-h] [-t {pdf,xml}] [-m {bart,presumm}] -cf N [N ...] -bhf N [N ...] -bf N [N ...] [-ns] [--output_xml_path PATH] PATH

Summarization of PDFs using BART

positional arguments:
  PATH                  path to input file

optional arguments:
  -h, --help            show this help message and exit
  -t {pdf,xml}, --file_type {pdf,xml}
                        type of file to summarize
  -m {bart,presumm}, --model {bart,presumm}
                        machine learning model choice
  -cf N [N ...], --chapter_heading_font N [N ...]
                        font of chapter titles
  -bhf N [N ...], --body_heading_font N [N ...]
                        font of headings within chapter
  -bf N [N ...], --body_font N [N ...]
                        font of body (the text you want to summarize)
  -ns, --no_summarize   do not run the summarization step
  --output_xml_path PATH
                        path to output XML file if `file_type` is `pdf`
```

Output of `python cmd_summarizer.py --help`

```bash
usage: cmd_summarizer.py [-h] -m {bart,presumm}

Summarization of text using CMD prompt

optional arguments:
  -h, --help            show this help message and exit
  -m {bart,presumm}, --model {bart,presumm}
                        machine learning model choice
```

Output of `python -m presumm.run_summarization --help`
```bash
usage: run_summarization.py [-h] --documents_dir DOCUMENTS_DIR [--summaries_output_dir SUMMARIES_OUTPUT_DIR] [--compute_rouge COMPUTE_ROUGE]
                            [--no_cuda NO_CUDA] [--batch_size BATCH_SIZE] [--min_length MIN_LENGTH] [--max_length MAX_LENGTH]
                            [--beam_size BEAM_SIZE] [--alpha ALPHA] [--block_trigram BLOCK_TRIGRAM]

optional arguments:
  -h, --help            show this help message and exit
  --documents_dir DOCUMENTS_DIR
                        The folder where the documents to summarize are located.
  --summaries_output_dir SUMMARIES_OUTPUT_DIR
                        The folder in wich the summaries should be written. Defaults to the folder where the documents are
  --compute_rouge COMPUTE_ROUGE
                        Compute the ROUGE metrics during evaluation. Only available for the CNN/DailyMail dataset.
  --no_cuda NO_CUDA     Whether to force the execution on CPU.
  --batch_size BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --min_length MIN_LENGTH
                        Minimum number of tokens for the summaries.
  --max_length MAX_LENGTH
                        Maixmum number of tokens for the summaries.
  --beam_size BEAM_SIZE
                        The number of beams to start with for each example.
  --alpha ALPHA         The value of alpha for the length penalty in the beam search.
  --block_trigram BLOCK_TRIGRAM
                        Whether to block the existence of repeating trigrams in the text generated by beam search.
```

### Notes

* `--file_type pdf` is only available on linux and requires `poppler-utils` to be installed

## PDF Structure

PDFs must be formatted in a specific way for this program to function. This program works with two levels of headings: `chapter` headings and `body` headings. `Chapter headings` contain many `body headings` and each body heading contains many lines of `body text`. If your PDF file is organized in this way and you can find unique font styles in the XML representation, then this program should work.

Sometimes italics or other stylistic fonts may be represented by separate font numbers. If this is the case simply run the command and pass in multiple font styles: `python main.py book.xml -cf 5 50 -bhf 23 34 60 -bf 11 132`.

## Meta

Hayden Housen – [haydenhousen.com](https://haydenhousen.com)

Distributed under the GPLv3 license. See the [LICENSE](LICENSE) for more information.

<https://github.com/HHousen>

PreSumm code extensively borrowed from [Hugging Face Transformers Library](https://github.com/huggingface/transformers/tree/master/examples/summarization).

## Contributing

All Pull Requests are greatly welcomed.

**Questions? Commends? Issues? Don't hesitate to open an [issue](https://github.com/HHousen/docsum/issues/new) and briefly describe what you are experiencing (with any error logs if necessary). Thanks.**

1. Fork it (<https://github.com/HHousen/docsum/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## To Do

* [ ] Make DocSum more robust to different PDF types (multi-layered headings)
* [ ] Implement other summarization techniques
* [ ] Implement automatic header detection ([Possibly this paper](https://arxiv.org/pdf/1809.01477.pdf))