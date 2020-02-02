import bart_sum
import os
import xml_processor
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Summarization of PDFs using BART')
parser.add_argument('file_path', metavar='PATH',
                    help='path to input file')
parser.add_argument('-t', '--file_type', default="xml", choices=["pdf", "xml"],
                    help='type of file to summarize')
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
args = parser.parse_args()

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
    # Load BART
    bart = bart_sum.load_bart()
    for chapter, content in tqdm(enumerate(book), total=len(book), desc="Chapter"):
        for heading in tqdm(content, desc="Heading"):
            document = content[heading]
            doc_length = len(document.split())
            min_len = int(doc_length/6)
            max_len_b = min_len+200
            content[heading] = bart_sum.summarize(bart, document, min_len=min_len, max_len_b=max_len_b)

# Save to file
with open("output.txt", "w") as file:
    for chapter, content in enumerate(book):
        file.write("Chapter " + str(chapter) + "\n" + "---------------------------\n")
        for heading in content:
            file.write(heading + "\n")
            file.write(content[heading] + "\n\n")