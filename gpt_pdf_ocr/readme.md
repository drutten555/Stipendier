## Readme

pdf_to_txt_gptocr: 
Takes a pdf, turns it to an image and sends it to openai api for ocr. Saves content of response as .txt

ocr_gptpolish:
Takes .txt, assumes it to be an imperfect ocr of a document. Sends it to openai api to be improved ("polished"). 

ocr_gptpolish_ensemble:
Takes directories of .txt's, assumed to be different imperfect ocrs of the same pdf. Sends them to openai api to recreate most probable original document. 

