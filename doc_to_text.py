import os
import pandas as pd
from PyPDF2 import PdfReader
import streamlit as st

# Loop through each file in the current directory
def extract_text(file_name):
    # st.text(file_name)
    if file_name.endswith('.pdf'):
        try:
            # Open the PDF file
            with open(file_name, 'rb') as file:
                # Initialize a PDF file reader
                pdf_reader = PdfReader(file)
                # Initialize text variable to store the content of the PDF
                text = ''

                # Iterate through each page in the PDF
                for page_num in range(len(pdf_reader.pages)):
                    # Extract text from the page
                    text += pdf_reader.pages[page_num].extract_text()
                    text = text.replace('\n',' ')
                # st.text(text)
                    
                # Create a new DataFrame with the file's title and text
                new_row = pd.DataFrame({'title': [file_name], 'text': [text]})

                return new_row
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    


