import datetime
import argparse
import bart_sum
import presumm.presumm as presumm

parser = argparse.ArgumentParser(description='Summarization of text using CMD prompt')
parser.add_argument('-m', '--model', choices=["bart", "presumm"], required=True,
                    help='machine learning model choice')
args = parser.parse_args()

print("> Summarizer: Loading Model")
if args.model == "bart":
    summarizer = bart_sum.BartSumSummarizer()
elif args.model == "presumm":
    summarizer = presumm.PreSummSummarizer()

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

        document = str(contents)
        print("> Summarizer: Document Created")


        doc_length = len(document.split())
        print("> Summarizer: Document Length: " + str(doc_length))

        min_len = int(doc_length/6)
        print("> Summarizer: min_len: " + str(min_len))
        max_len_b = min_len+200
        print("> Summarizer: max_len_b: " + str(max_len_b))

        transcript_summarized = summarizer.summarize_string(document, min_len=min_len, max_len_b=max_len_b)
        with open("summarized.txt", 'a+') as file:
            file.write("\n" + str(datetime.datetime.now()) + ":\n")
            file.write(transcript_summarized + "\n")

except KeyboardInterrupt:
    print("Exiting...")