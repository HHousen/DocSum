# DocSum
> A tool to automatically summarize documents using the BART Machine Learning Model.

BART ([BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)) is the state-of-the-art in text summarization as of 02/02/2020. It is a "sequence-to-sequence model trained with denoising as pretraining objective" ([Documentation & Examples](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.md)).

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
├── bart_sum_commandline.py
├── bart_sum.py
├── environment.yml
├── LICENSE
├── main.py
├── README.md
└── xml_processor.py
```

## Usage
Output of `python main.py --help`:
```
usage: main.py [-h] [-t {pdf,xml}] -cf N [N ...] -bhf N [N ...] -bf N [N ...] [-ns] [--output_xml_path PATH] PATH

Summarization of PDFs using BART

positional arguments:
  PATH                  path to input file

optional arguments:
  -h, --help            show this help message and exit
  -t {pdf,xml}, --file_type {pdf,xml}
                        type of file to summarize
  -cf N [N ...], --chapter_heading_font N [N ...]
                        font of chapter titles
  -bhf N [N ...], --body_heading_font N [N ...]
                        font of headings within chapter
  -bf N [N ...], --body_font N [N ...]
                        font of body (the text you want to summarize)
  -ns, --no_summarize   do not run the summarization step
  --output_xml_path PATH
                        path to output XML file if `file_type` is `pdf
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