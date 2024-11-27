from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv

root_directory = "./output"  
llm_analyzed_directory = "./LLM-analyzed"
os.makedirs(llm_analyzed_directory, exist_ok=True)

prompt = (
    "What are the electricity rates for residential customer? Also mention if there's any EV and Time of Day tariffs?"
)
load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

# Helper function to process each PDF file
def process_pdf(pdf_path, pdf_assistant, utility_name):
    # Create thread
    thread = client.beta.threads.create()

    # Upload file
    file = client.files.create(file=open(pdf_path, "rb"), purpose="assistants")

    # Create assistant prompt for the specific file
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        attachments=[
            Attachment(
                file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
            )
        ],
        content=prompt,
    )

    # Run thread and poll the response
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
    )

    if run.status != "completed":
        raise Exception("Run failed:", run.status)

    # Get the result messages
    messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
    messages = [message for message in messages_cursor]

    # Extract response text and save to file
    res_txt = messages[0].content[0].text.value
    output_file_path = os.path.join(llm_analyzed_directory, f"{utility_name}_analysis.txt")
    with open(output_file_path, "w") as f:
        f.write(res_txt)
    print(f"Analysis saved for utility: {utility_name}")

# Creating PDF assistant
pdf_assistant = client.beta.assistants.create(
    model="gpt-4o",
    description="An assistant to extract the contents of PDF files.",
    tools=[{"type": "file_search"}],
    name="PDF assistant",
)

# Walk through all folders and process PDFs
all_pdf_files = []
for utility_folder_name in os.listdir(root_directory):
    utility_folder_path = os.path.join(root_directory, utility_folder_name)

    # Check if it's a directory and contains any PDF files
    if os.path.isdir(utility_folder_path):
        pdf_files = [
            file
            for file in os.listdir(utility_folder_path)
            if file.lower().endswith(".pdf")
        ]

        if pdf_files:
            for pdf_file in pdf_files:
                pdf_path = os.path.join(utility_folder_path, pdf_file)
                all_pdf_files.append((pdf_path, utility_folder_name))

# Process all PDFs with progress bar
for pdf_path, utility_folder_name in tqdm(all_pdf_files, desc="Processing PDFs"):
    try:
        process_pdf(pdf_path, pdf_assistant, utility_folder_name)
        # Adding a delay to avoid hitting rate limits
        time.sleep(5)
    except Exception as e:
        print(f"Failed to analyze {pdf_path}: {e}")
