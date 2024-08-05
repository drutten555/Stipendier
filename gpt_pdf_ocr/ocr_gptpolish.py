#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:05:30 2024

Author: Filip Berntsson
"""

import os
import chardet
from openai import OpenAI

###############################################################################
##- Helper functions --------------------------------------------------------##
###############################################################################

def get_api_key(api_path):
    """
    Read the API key from a specified file.

    Args:
        api_path (str): The path to the file containing the API key.

    Returns:
        str: The API key.
    """
    with open(api_path, 'r') as file:
        return file.read().strip()

def get_encoding(file_path):
    """
    Detect the encoding of a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The detected encoding of the file.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def get_ocr(file_path):
    """
    Get the OCR content from a text file as a string.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The OCR content of the file.
    """
    encoding = get_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
    return content

def construct_messages(ocr):
    """
    Construct messages for GPT-4 based on OCR content.

    Args:
        ocr (str): The OCR content.

    Returns:
        list: A list of messages formatted for GPT-4 input.
    """
    system_prompt = (
        "Du har en mycket viktig uppgift!\n\n"
        "#### DIN UPPGIFT ####\n"
        "En mjukvara har producerat följande OCR av en pdf.\n"
        "Ditt jobb är följande: Med hjälp av den presenterade OCRen ska du återskapa texten i dokumentet, så ordagrannt som möjligt.\n\n"
        "#### DETALJER #### \n\n"
        "Du är strikt förbjuden att ändra på innehållet i dokumentet och det är av yttersta vikt att det du returnerar ska vara så nära originalet du kan förmå. "
        "Om det finns tecken eller symboler du ej kan avläsa, ersätt dem med en asterix: '*'\n\n"
    )

    # Create initial messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Append the OCR in a nice format
    messages.append({"role": "user", "content": f"#### OCR-transkription: #### \n{ocr}\n\n{ocr}"})
    
    return messages

def generate_response(client, model, messages, max_tokens):
    """
    Generate a response from GPT-4 based on the given messages.

    Args:
        client (OpenAI): The OpenAI client.
        model (str): The model to use (e.g., "gpt-4o").
        messages (list): The messages to send to the model.
        max_tokens (int): The maximum number of tokens for the response.

    Returns:
        dict: The response from GPT-4.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return response

def main():
    """
    To process a .txt OCR. Generate polished text using GPT-4, 
    and save the polished text to a specified directory.
    """
    # Paths and parameters
    ## DIR_PATH points to a folder in which there are 2 more folders.
    ## 1) ABBYY_OCR. This folder has one .txt file for each stiftelse/gmf
    ## 2) PROCESSED_OCR This folder is preferably empty. Here the ocrs that gpt
    ##    has polished will end up.
    DIR_PATH = '/Users/filipberntsson/Documents/Work/P1/transcriptions'
    API_PATH = '/Users/filipberntsson/Documents/Work/misc/api_key.txt'
    MODEL = "gpt-4o-mini"  # Consider changing to gpt-4o when running for real
    MAX_TOKENS = 200  # Change to 3-4k when running for real

    # Initialize OpenAI client
    client = OpenAI(api_key=get_api_key(API_PATH))

    # Define paths for input and output directories
    ABBYY_PATH = os.path.join(DIR_PATH, 'ABBYY_OCR')
    OUT_PATH = os.path.join(DIR_PATH, 'PROCESSED_OCR')

    # Check if the main directory exists
    if not os.path.exists(DIR_PATH):
        print(f"Error: The directory {DIR_PATH} does not exist.")
        return
    
    # Check if ABBYY_OCR directory exists
    if not os.path.exists(ABBYY_PATH):
        print(f"Error: The directory {ABBYY_PATH} does not exist.")
        return

    # Ensure the output directory exists
    os.makedirs(OUT_PATH, exist_ok=True)

    # Process each text file in the ABBYY_OCR directory
    file_paths = [os.path.join(ABBYY_PATH, f) for f in os.listdir(ABBYY_PATH) if os.path.isfile(os.path.join(ABBYY_PATH, f))]
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        ocr = get_ocr(file_path)
        messages = construct_messages(ocr)
        response = generate_response(client, MODEL, messages, MAX_TOKENS)
        content = response.choices[0].message.content  # Extract the actual response content
        
        # Construct the save path for the processed text
        save_path = os.path.join(OUT_PATH, os.path.basename(file_path).replace("ABBYY", "PROCESSED"))
        
        # Save the polished text to the output directory
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Saved processed file to: {save_path}")

    print("DONE")

if __name__ == "__main__":
    main()