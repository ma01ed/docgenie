import os
import shutil
from datetime import datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from transformers import pipeline


app = FastAPI()

# create a directory to store uploaded files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# function to generate a unique filename for uploaded files
def get_unique_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename, ext = os.path.splitext(filename)
    return f"{timestamp}_{basename}{ext}"


# function to delete files that are older than 48 hours
def delete_old_files():
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.stat(filepath).st_mtime < (datetime.now() - timedelta(hours=48)).timestamp():
            os.remove(filepath)


# define the file upload route
@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    # generate a unique filename for the uploaded file
    filename = get_unique_filename(file.filename)
    # save the file to the uploads directory
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
        shutil.copyfileobj(file.file, f)
    # delete files that are older than 48 hours
    delete_old_files()
    # return a success message with the unique filename
    return {"message": "File uploaded successfully", "filename": filename}


# define the inference route
@app.post("/answer_question/")
async def answer_question(question: str = Form(...), filename: str = Form(...)):
    # load the file from the uploads directory
    filepath = os.path.join(UPLOAD_DIR, filename)
    # check if the file exists
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    
    
    # load the file contents into memory
    with open(filepath, "r") as f:
        file_contents = f.read()
    # use the Hugging Face pipeline to answer the question
    nlp = pipeline("question-answering")
    answer = nlp({"question": question, "context": file_contents})
    return {"answer": answer["answer"]}

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

nlp(
    "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
    "What is the invoice number?"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

