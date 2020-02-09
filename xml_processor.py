from collections import OrderedDict
from unidecode import unidecode
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_xml(xml_path):
    """Obtain representation of XML file"""
    xml_root = ET.parse(xml_path).getroot()
    return xml_root

def get_chapter_page_numbers(xml_root, fonts, closeness=3):
    """
    Create list of chapter page numbers.
    `closeness` determines how far pages need to be apart in order to be considered a new chapter
    """
    chapter_start_pages = list()
    for page in xml_root:
        page_num = int(page.attrib['number'])
        for item in page:
            if item.tag == "text":
                if item.attrib['font'] in fonts: # chapter detection
                    chapter_start_pages.append(page_num)
                    break

    # Clean chapter_start_pages by removing page numbers that are too close together
    previous_number = 0
    for page_number in chapter_start_pages:
        if previous_number+closeness > page_number:
            chapter_start_pages.remove(previous_number)
        previous_number = page_number
    
    return chapter_start_pages

def process(xml_root, chapter_start_pages, heading_fonts, body_fonts):
    content = OrderedDict()
    heading = ""
    first_body = True
    book = list()
    last_chapter_num = 1
    for page in tqdm(xml_root, desc="Page"):
        current_page_num = int(page.attrib['number'])
        # Get current chapter based on page number
        for idx, page_number in enumerate(chapter_start_pages):
            # If the current page number is less than or equal to every chapter start page number
            if current_page_num+1 <= page_number:
                chapter_num = idx+1
                break
            else:
                chapter_num = 0

        # If the chapter number has changed since the last page then save content and reset        
        if last_chapter_num != chapter_num:
            # print("last_chapter_num: " + str(last_chapter_num) + "          chapter_num: " + str(chapter_num))
            book.append(content)
            content = OrderedDict()
            first_body = True

        # Set last chapter number to the current chapter number
        last_chapter_num = chapter_num

        for item in page:
            if item.tag == "text":
                # If item is a heading
                if item.attrib['font'] in heading_fonts:
                    first_body = True
                    heading += item[0].text
                # If item is body text
                if item.attrib['font'] in body_fonts:
                    # If this is the first body after the heading then set the `current_heading` and initialize the `content` section
                    if first_body:
                        current_heading = heading.replace('\n', ' ').strip()
                        if current_heading == "":
                            current_heading = "Unknown"
                        current_heading = unidecode(current_heading)
                        content[current_heading] = ""
                        heading = ""
                    first_body = False
                    
                    # convert unicode to ascii
                    text = unidecode(item.text)
                    # strip whitespace and replace newlines
                    text = text.strip().replace('\n', ' ').replace('\r', ' ')
                    # remove dashes from lines that end in dashes
                    if text[-1:] == "-":
                        text = text[-1:]
                    # add space after each line
                    text += " "

                    # Store line of text in `content` dictionary under `current_heading`
                    content[current_heading] += text
    return book


