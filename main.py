import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import pdfplumber
from googleapiclient.discovery import build
import time

class TariffExtractor:
    def __init__(self):
        """Initialize the extractor with API configurations."""
        # Load environment variables
        load_dotenv()

        # OpenAI API setup
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env file")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)

        # Google Custom Search API setup
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.custom_search_id = os.getenv('GOOGLE_CSE_ID')
        if not self.google_api_key or not self.custom_search_id:
            raise ValueError("Google API key or Custom Search Engine ID not found in .env file")

        try:
            self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
        except Exception as e:
            logging.error("Error initializing Google API Client: %s", e)
            raise

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tariff_extractor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def download_pdf(self, pdf_url: str) -> Optional[str]:
        """Download PDF with custom headers and a fallback to no headers."""
        max_retries = 3
        retry_delay = 5
        timeout = 60

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
            'Accept': 'application/pdf',
            'Referer': pdf_url.split('/')[2]  # Use the domain as the Referer
        }

        for attempt in range(max_retries):
            try:
                # First attempt: With headers
                self.logger.info(f"Attempt {attempt + 1}: Downloading PDF with headers from {pdf_url}")
                response = requests.get(pdf_url, headers=headers, timeout=timeout)
                response.raise_for_status()

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(response.content)
                temp_file.close()

                self.logger.info(f"Successfully downloaded PDF: {temp_file.name}")
                return temp_file.name
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Download failed with headers (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    break

            try:
                # Fallback: Without headers
                self.logger.info(f"Attempt {attempt + 1}: Retrying download without headers for {pdf_url}")
                response = requests.get(pdf_url, timeout=timeout)
                response.raise_for_status()

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(response.content)
                temp_file.close()

                self.logger.info(f"Successfully downloaded PDF: {temp_file.name}")
                return temp_file.name
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Download failed without headers (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                self.logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        self.logger.error(f"Failed to download PDF after {max_retries} attempts: {pdf_url}")
        return None


    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text and tables from a PDF using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page_number, page in enumerate(pdf.pages, start=1):
                    self.logger.info(f"Processing page {page_number}")

                    # Extract main text from the page
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_number}:\n{page_text}\n"

                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        text += "\nRATE SCHEDULE TABLES:\n"
                        for table in tables:
                            for row in table:
                                # Ensure all cells are strings, replace None with empty string
                                row = [str(cell) if cell is not None else "" for cell in row]
                                text += " | ".join(row) + "\n"
                    else:
                        self.logger.info(f"No tables found on page {page_number}")

                self.logger.debug(f"Extracted text sample (first 500 chars): {text[:500]}")
                return text

        except Exception as e:
            self.logger.error(f"Error extracting text and tables from PDF: {str(e)}")
            return None

    
    def extract_tariff_info(self, pdf_text: str, discom_name: str, year: int) -> Dict:
        """Extract residential/LT tariff information using GPT."""
        try:
            prompt = f"""Analyze this electricity tariff document for {discom_name} and extract ONLY the residential/domestic/LT customer tariff information.

Focus specifically on finding and extracting data from the Rate tables that typically has columns like
Consumption Slab (kWh) | Fixed/Demand Charge | Energy Charge (Rs./kWh)

Return the data in this exact JSON structure:
{{
    "residential_tariffs": {{
        "fixed_charges": [...],
        "energy_charges": [...],
        "additional_charges": [...],
        "conditions": [...],
        "effective_date": "string"
    }}
}}

Document text:
{pdf_text}"""

            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a tariff analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            parsed_response = json.loads(response.choices[0].message.content)
            if not parsed_response.get('residential_tariffs'):
                self.logger.warning(f"No residential tariff data found for {discom_name}")
            return parsed_response

        except Exception as e:
            self.logger.error(f"Error in GPT extraction: {str(e)}")
            return {"error": "Failed to extract tariff information", "details": str(e)}

    def process_discom(self, discom_info: Dict) -> Dict:
        """Process workflow for a single DISCOM."""
        try:
            self.logger.info(f"Searching tariff documents for {discom_info['discom']}")

            search_variations = [
                '2024 residential electricity tariff rates schedule',
                '2024 domestic consumer tariff pdf'
            ]

            best_result = None
            for search_term in search_variations:
                search_query = f'site:{discom_info["website"]} {search_term} filetype:pdf'

                try:
                    result = self.google_service.cse().list(
                        q=search_query,
                        cx=self.custom_search_id,
                        num=5
                    ).execute()

                    if 'items' in result:
                        for item in result['items']:
                            if item['link'].endswith('.pdf'):
                                best_result = item
                                break

                except Exception as e:
                    self.logger.warning(f"Search variation failed: {str(e)}")
                    continue

            if not best_result:
                return {'discom': discom_info['discom'], 'error': 'No document found'}

            pdf_path = self.download_pdf(best_result['link'])
            if not pdf_path:
                return {'discom': discom_info['discom'], 'error': 'PDF download failed'}

            content = self.extract_text_from_pdf(pdf_path)
            if not content:
                return {'discom': discom_info['discom'], 'error': 'Content extraction failed'}

            return self.extract_tariff_info(content, discom_info['discom'], 2024)

        except Exception as e:
            self.logger.error(f"Error processing {discom_info['discom']}: {str(e)}")
            return {'discom': discom_info['discom'], 'error': str(e)}

def main():
    try:
        extractor = TariffExtractor()
        discoms = [
            # {"state": "Delhi", "discom": "Tata Power Delhi", "website": "www.tatapower-ddl.com"},
            # {"state": "Delhi", "discom": "BSES Rajdhani", "website": "www.bsesdelhi.com"},
            # {"state": "Maharashtra", "discom": "MSEDCL", "website": "www.mahadiscom.in"},
            {"state": "Karnataka", "discom": "BESCOM", "website": "bescom.org"},
        ]

        results = []
        for discom in discoms:
            result = extractor.process_discom(discom)
            results.append(result)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"tariff_data.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logging.error(f"Main execution error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
