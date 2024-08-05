#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:56:03 2024

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

def get_ocrs(s_path):
    """
    Get a list of OCR contents from all text files in a directory.

    Args:
        s_path (str): The path to the directory containing text files.

    Returns:
        list: A list of OCR contents.
    """
    file_paths = [os.path.join(s_path, f) for f in os.listdir(s_path) if f.endswith('.txt')]
    ocrs = [get_ocr(f_path) for f_path in file_paths]
    return ocrs

def construct_messages(ocrs):
    """
    Construct messages for GPT-4 based on OCR contents.

    Args:
        ocrs (list): A list of OCR contents.

    Returns:
        list: A list of messages formatted for GPT-4 input.
    """
    system_prompt = (
        "Du har en mycket viktig uppgift!\n\n"
        "#### DIN UPPGIFT ####\n"
        "Din uppgift är att ordagrant återskapa texten från ett inscannat dokument. "
        "För att göra detta kommer du få tillgång till en eller flera OCR-transkriptioner av dokumentet.\n\n"
        "#### DETALJER #### \n\n"
        "Analysera de tillgängliga OCR-transkriptionerna och konstruera baserat på dem den "
        "mest exakta ordagranna transkriptionen av originaldokumentet. "
        "Ta hänsyn till skillnader mellan transkriptionerna, såväl som löptextens innebörd och sammanhang.\n\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    for i, ocr in enumerate(ocrs, start=1):
        messages.append({"role": "user", "content": f"#### OCR-transkription {i}: #### \n{ocr}\n\n"})
    
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

def save_response(content, s_name, save_dir):
    """
    Save the GPT-4 response to a text file.

    Args:
        content (str): The content to save.
        s_name (str): The name of the subdirectory.
        save_dir (str): The directory to save the text file.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{s_name}-ensemble.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_subdirectory(client, model, max_tokens, s_path, save_dir):
    """
    Process a single subdirectory to generate and save the transcription.
    Assumes all documents in the s_path are .txt files, ocrs of the same pdf.
    Sends these to the openai api and asks it to reproduce the original document
    based on this information. 

    Args:
        client (OpenAI): The OpenAI client.
        model (str): The model to use (e.g., "gpt-4o").
        max_tokens (int): The maximum number of tokens for the response.
        s_path (str): The path to the subdirectory.
        save_dir (str): The directory to save the transcription.
    """
    ocrs = get_ocrs(s_path)
    messages = construct_messages(ocrs)
    response = generate_response(client, model, messages, max_tokens)
    content = response.choices[0].message.content
    s_name = os.path.basename(s_path)
    save_response(content, s_name, save_dir)

def main():
    """
    Main function to process all subdirectories and generate transcriptions.
    """
    # Path to the file containing the API key. This should be updated to the actual path on your system.
    API_PATH = '{YOUR PATH HERE}'
    
    # Path to the directory containing subfolder with OCR text files.
    # One folder for each pdf-document. The idea here is that several different
    # OCRS are available for each pdf, and this method asks gpt to take in all
    # of these and reproduce the most likely original document. 
    # Update this to the actual directory path.
    DIR_PATH = '{YOUR PATH HERE}'
    
    # Path to the directory where the transcriptions will be saved. Update this to the desired save location.
    SAVE_DIR = '{YOUR PATH HERE}'
    MODEL = "gpt-4o"
    MAX_TOKENS = 1000  # Adjust to your needs

    client = OpenAI(api_key=get_api_key(API_PATH))

    # Find all subdirectories in DIR_PATH
    subdirectories = [os.path.join(DIR_PATH, d) for d in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH, d))]

    for s_path in subdirectories:
        print(f"Processing directory: {s_path}")
        process_subdirectory(client, MODEL, MAX_TOKENS, s_path, SAVE_DIR)
    
    print("DONE")

if __name__ == "__main__":
    main()
