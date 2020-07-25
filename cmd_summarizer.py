import datetime
import argparse
import bart_sum
import logging
import presumm.presumm as presumm

logger = logging.getLogger(__name__)

def do_summarize(contents):
    document = str(contents)
    logger.info("Document Created")


    doc_length = len(document.split())
    logger.info("Document Length: " + str(doc_length))

    min_length = int(doc_length/6)
    logger.info("min_length: " + str(min_length))
    max_length = min_length+200
    logger.info("max_length: " + str(max_length))

    transcript_summarized = summarizer.summarize_string(document, min_length=min_length, max_length=max_length)
    with open("summarized.txt", 'a+') as file:
        file.write("\n" + str(datetime.datetime.now()) + ":\n")
        file.write(transcript_summarized + "\n")

parser = argparse.ArgumentParser(description='Summarization of text using CMD prompt')
parser.add_argument('-m', '--model', choices=["bart", "presumm"], required=True,
                    help='machine learning model choice')
parser.add_argument('--bart_checkpoint', default=None, type=str, metavar='PATH',
                    help='[BART Only] Path to optional checkpoint. Semsim is better model but will use more memory and is an additional 5GB download. (default: none, recommended: semsim)')
parser.add_argument('--bart_state_dict_key', default='model', type=str, metavar='PATH',
                    help='[BART Only] model state_dict key to load from pickle file specified with --bart_checkpoint (default: "model")')
parser.add_argument('--bart_fairseq', action='store_true',
                    help='[BART Only] Use fairseq model from torch hub instead of huggingface transformers library models. Can not use --bart_checkpoint if this option is supplied.')
parser.add_argument('--text', default=None, type=str,
                    help='Optional text to summarize if you cannot paste it using an interactive shell.')
parser.add_argument("-l", "--log", dest="logLevel", default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="Set the logging level (default: 'Info').")
args = parser.parse_args()

logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s> %(message)s", level=logging.getLevelName(args.logLevel))

logger.info("Loading Model")
if args.model == "bart":
    summarizer = bart_sum.BartSumSummarizer(checkpoint=args.bart_checkpoint,
                                            state_dict_key=args.bart_state_dict_key,
                                            hg_transformers=(not args.bart_fairseq))
elif args.model == "presumm":
    summarizer = presumm.PreSummSummarizer()

if args.text:
    do_summarize(args.text)
else:
    try:
        while True:
            print("Enter/Paste your content. Ctrl-D or Ctrl-Z (windows) to save it. Ctrl-C to exit.")
            contents = ""
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                contents += (line.strip()+ " ")

            do_summarize(contents)

    except KeyboardInterrupt:
        print("Exiting...")