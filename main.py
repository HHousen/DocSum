import bart_sum
import presumm.presumm as presumm
import os
import xml_processor
import argparse
import logging
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Summarization of PDFs using BART')
parser.add_argument('file_path', metavar='PATH',
                    help='path to input file')
parser.add_argument('-t', '--file_type', default="xml", choices=["pdf", "xml"],
                    help='type of file to summarize')
parser.add_argument('-m', '--model', default="bart", choices=["bart", "presumm"],
                    help='machine learning model choice')
parser.add_argument('--bart_checkpoint', default=None, type=str, metavar='PATH',
                    help='[BART Only] Path to optional checkpoint. Semsim is better model but will use more memory and is an additional 5GB download. (default: none, recommended: semsim)')
parser.add_argument('--bart_state_dict_key', default='model', type=str, metavar='PATH',
                    help='[BART Only] model state_dict key to load from pickle file specified with --bart_checkpoint (default: "model")')
parser.add_argument('--bart_fairseq', action='store_true',
                    help='[BART Only] Use fairseq model from torch hub instead of huggingface transformers library models. Can not use --bart_checkpoint if this option is supplied.')
parser.add_argument('-cf', '--chapter_heading_font', nargs='+', default=0, type=int, metavar='N', required=True,
                    help='font of chapter titles')
parser.add_argument('-bhf', '--body_heading_font', nargs='+', default=0, type=int, metavar='N', required=True,
                    help='font of headings within chapter')
parser.add_argument('-bf', '--body_font', nargs='+', default=0, type=int, metavar='N', required=True,
                    help='font of body (the text you want to summarize)')
parser.add_argument('-ns', '--no_summarize', action='store_true',
                    help='do not run the summarization step')
parser.add_argument('--output_xml_path', metavar='PATH',
                    help='path to output XML file if `file_type` is `pdf`')
parser.add_argument("-l", "--log", dest="logLevel", default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="Set the logging level (default: 'Info').")
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s> %(message)s", level=logging.getLevelName(args.logLevel))

if args.file_type == "pdf":
    if not args.output_xml_path:
        args.output_xml_path = "output.xml"
    os.system('pdftohtml ' + args.file_path + '.pdf -i -s -c -xml ' + args.output_xml_path)
    args.file_path = args.output_xml_path

args.chapter_heading_font = [str(i) for i in args.chapter_heading_font]
args.body_heading_font = [str(i) for i in args.body_heading_font]
args.body_font = [str(i) for i in args.body_font]

xml_root = xml_processor.parse_xml(args.file_path)
chapter_start_pages = xml_processor.get_chapter_page_numbers(xml_root, fonts=args.chapter_heading_font)
book = xml_processor.process(xml_root, chapter_start_pages, heading_fonts=args.body_heading_font, body_fonts=args.body_font)

# Summarize each section of the `book` list
if not args.no_summarize:
    if args.model == "bart":
        summarizer = bart_sum.BartSumSummarizer(checkpoint=args.bart_checkpoint,
                                                state_dict_key=args.bart_state_dict_key,
                                                hg_transformers=(not args.bart_fairseq))
    elif args.model == "presumm":
        summarizer = presumm.PreSummSummarizer()
    
    for chapter, content in tqdm(enumerate(book), total=len(book), desc="Chapter"):
        for heading in tqdm(content, desc="Heading"):
            document = content[heading]
            doc_length = len(document.split())
            min_length = int(doc_length/6)
            max_length = min_length+200
            content[heading] = summarizer.summarize_string(document, min_length=min_length, max_length=max_length)

# Save to file
with open("output.txt", "w") as file:
    for chapter, content in enumerate(book):
        file.write("Chapter " + str(chapter) + "\n" + "---------------------------\n")
        for heading in content:
            file.write(heading + "\n")
            file.write(content[heading] + "\n\n")