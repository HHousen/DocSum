import bart_sum
import datetime

print("> BART Summarizer: Loading BART Model")
bart = bart_sum.load_bart()

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
        print("> BART Summarizer: Document Created")


        doc_length = len(document.split())
        print("> BART Summarizer: Document Length: " + str(doc_length))

        min_len = int(doc_length/6)
        print("> BART Summarizer: min_len: " + str(min_len))
        max_len_b = min_len+200
        print("> BART Summarizer: max_len_b: " + str(max_len_b))

        transcript_summarized = bart_sum.summarize(bart, document, min_len=min_len, max_len_b=max_len_b)
        with open("summarized.txt", 'a+') as file:
            file.write("\n" + str(datetime.datetime.now()) + ":\n")
            file.write(transcript_summarized + "\n")

except KeyboardInterrupt:
    print("Exiting...")