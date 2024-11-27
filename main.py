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
import pytesseract
from PIL import Image
import io

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
        """Extract text and tables from a PDF, with OCR fallback for scanned pages."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                max_pages = min(len(pdf.pages), 20)  # Limit to the first 20 pages
                for page_number, page in enumerate(pdf.pages[:max_pages], start=1):
                    self.logger.info(f"Processing page {page_number}")

                    # Try to extract text directly
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_number}:\n{page_text}\n"
                    else:
                        self.logger.warning(f"No text found on page {page_number}, attempting OCR...")

                        # Perform OCR on the page's image
                        page_image = page.to_image()
                        pil_image = page_image.original  # Get the PIL Image
                        ocr_text = pytesseract.image_to_string(pil_image, lang='eng')
                        text += f"Page {page_number} (OCR):\n{ocr_text}\n"

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
    Consumption Slab (kWh) | Fixed/Demand Charge | Energy Charge (Rs./kWh) or Wheeling Charge | etc.

    Return the data in this exact JSON structure with residential rates for different slabs.

    Document text:
    {pdf_text}"""

            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a tariff analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            # Log the raw response for debugging
            self.logger.debug(f"Raw OpenAI response: {response}")

            # Check if response contains choices
            if not response.choices or len(response.choices) == 0:
                self.logger.error(f"No choices returned in OpenAI response: {response}")
                return {"error": "No choices returned in OpenAI response", "raw_response": str(response)}

            # Get the first choice and validate message.content
            choice = response.choices[0]
            if not choice.message or not choice.message.content:
                self.logger.error(f"Invalid or empty message content in OpenAI response: {choice}")
                return {"error": "Invalid or empty message content", "details": str(choice)}
            
            print(choice.message.content)

            # For every utility, the rate table is different. So, we need to extract the rate table from the response. 
            # We will do that later, for now, we will just return and save the response as is in the utility specific folder
            
            # Save the response in the utility-specific folder
            utility_folder = os.path.join("output", discom_name.replace(" ", "_"))
            os.makedirs(utility_folder, exist_ok=True)
            response_file = os.path.join(utility_folder, f"tariff_response.json")
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(choice.message.content, f, indent=2, ensure_ascii=False)

            

            return {"response": choice.message.content}

        except Exception as e:
            self.logger.error(f"Error in GPT extraction: {str(e)}")
            return {"error": "Failed to extract tariff information", "details": str(e)}

    def process_discom(self, discom_info: Dict) -> Dict:
        """Process workflow for a single DISCOM."""
        try:
            self.logger.info(f"Searching tariff documents for {discom_info['discom']}")

            # Create a utility-specific folder
            utility_folder = os.path.join("output", discom_info["discom"].replace(" ", "_"))
            os.makedirs(utility_folder, exist_ok=True)

            # Search variations to try
            search_variations = [
                '2024 residential electricity tariff rates schedule',
                '2024 domestic consumer tariff pdf'
            ]

            best_result = None
            explored_urls = []  # To track all explored URLs
            for search_term in search_variations:
                search_query = f'site:{discom_info["website"]} {search_term} filetype:pdf'

                self.logger.info(f"Trying search query: {search_query}")
                try:
                    result = self.google_service.cse().list(
                        q=search_query,
                        cx=self.custom_search_id,
                        num=5
                    ).execute()

                    if 'items' in result:
                        for item in result['items']:
                            explored_urls.append(item['link'])  # Log the explored URL
                            if item['link'].endswith('.pdf'):
                                best_result = item
                                break

                    if best_result:
                        break

                except Exception as e:
                    self.logger.warning(f"Search variation failed: {str(e)}")
                    continue

            if not best_result:
                return {
                    'discom': discom_info['discom'],
                    'state': discom_info['state'],
                    'error': 'No document found',
                    'explored_urls': explored_urls
                }

            # Download the best result PDF
            pdf_path = self.download_pdf(best_result['link'])
            if not pdf_path:
                return {
                    'discom': discom_info['discom'],
                    'state': discom_info['state'],
                    'error': 'PDF download failed',
                    'explored_urls': explored_urls,
                    'best_url': best_result['link']
                }

            # Save the downloaded PDF in the utility-specific folder
            pdf_filename = os.path.join(utility_folder, os.path.basename(best_result['link']))
            os.rename(pdf_path, pdf_filename)
            self.logger.info(f"PDF saved to {pdf_filename}")

            # Extract content from the PDF
            content = self.extract_text_from_pdf(pdf_filename)
            if not content:
                return {
                    'discom': discom_info['discom'],
                    'state': discom_info['state'],
                    'error': 'Content extraction failed',
                    'explored_urls': explored_urls,
                    'best_url': best_result['link']
                }

            # Extract tariff information
            # tariff_data = self.extract_tariff_info(content, discom_info['discom'], 2024)

            return {
                'discom': discom_info['discom'],
                'state': discom_info['state'],
                'source_url': best_result['link'],
                'source_title': best_result['title'],
                'source_type': best_result['type'],
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'tariff_data': content,
                'explored_urls': explored_urls
            }

        except Exception as e:
            self.logger.error(f"Error processing {discom_info['discom']}: {str(e)}")
            return {
                'discom': discom_info['discom'],
                'state': discom_info['state'],
                'error': str(e),
                'explored_urls': explored_urls
            }

        



def main():
    try:
        extractor = TariffExtractor()
        # Create list of dictionaries containing DISCOM information for all states
        discoms = [
            # North Zone
            {"state": "Himachal Pradesh", "discom": "Himachal Pradesh State Electricity Board Limited", "website": "hpseb.in"},
            {"state": "Delhi", "discom": "Tata Power Delhi Distribution Limited", "website": "tatapower-ddl.com"},
            {"state": "Delhi", "discom": "BSES Rajdhani Power Limited", "website": "bsesdelhi.com"},
            {"state": "Delhi", "discom": "BSES Yamuna Power Limited", "website": "bsesdelhi.com"},
            {"state": "Delhi", "discom": "New Delhi Municipal Corporation", "website": "ndmc.gov.in"},
            {"state": "Haryana", "discom": "Uttar Haryana Bijli Vitran Nigam", "website": "uhbvn.org.in"},
            {"state": "Haryana", "discom": "Dakshin Haryana Bijli Vitran Nigam", "website": "dhbvn.org.in"},
            {"state": "Uttarakhand", "discom": "Uttarakhand Power Corporation Limited", "website": "upcl.org"},
            {"state": "Punjab", "discom": "Punjab State Power Corporation Limited", "website": "pspcl.in"},
            {"state": "Uttar Pradesh", "discom": "Purvanchal Vidyut Vitran Nigam Ltd.", "website": "puvvnl.up.nic.in"},
            {"state": "Uttar Pradesh", "discom": "Paschimanchal Vidyut Vitran Nigam Limited", "website": "pvvnl.up.nic.in"},
            {"state": "Uttar Pradesh", "discom": "Madhyanchal Vidyut Vitran Nigam Limited", "website": "mvvnl.in"},
            {"state": "Uttar Pradesh", "discom": "Dhakshinachal Vidyut Vitran Nigam Limited", "website": "dvvnl.org"},
            {"state": "Chandigarh", "discom": "Electricity Department, UT of Chandigarh", "website": "chdengineering.gov.in"},
            {"state": "Jammu & Kashmir", "discom": "Power Development Department", "website": "jkpdd.gov.in"},

            # North-East & East Zone
            {"state": "Manipur", "discom": "Manipur State Power Distribution Company Ltd", "website": "mspdcl.in"},
            {"state": "Arunachal Pradesh", "discom": "Department of Power, Arunachal Pradesh", "website": "power.arunachal.gov.in"},
            {"state": "Nagaland", "discom": "Department of Power, Nagaland", "website": "power.nagaland.gov.in"},
            {"state": "Sikkim", "discom": "Sikkim Power Development Corporation Limited", "website": "spdcl.sikkim.gov.in"},
            {"state": "Meghalaya", "discom": "Meghalaya Energy Distribution Corporation Limited", "website": "meecl.nic.in"},
            {"state": "Mizoram", "discom": "Power & Electricity Department, Government of Mizoram", "website": "power.mizoram.gov.in"},
            {"state": "Bihar", "discom": "North Bihar Power Distribution Company Limited", "website": "nbpdcl.in"},
            {"state": "Bihar", "discom": "South Bihar Power Distribution Company Limited", "website": "sbpdcl.in"},
            {"state": "Assam", "discom": "Assam Power Distribution Company Limited", "website": "apdcl.org"},
            {"state": "Tripura", "discom": "Tripura State Electricity Corporation Limited", "website": "tsecl.in"},

            # South Zone
            {"state": "Kerala", "discom": "Kerala State Electricity Board Limited", "website": "kseb.in"},
            {"state": "Karnataka", "discom": "Chamundeshwari Electricity Supply Corporation Limited", "website": "cescmysore.org"},
            {"state": "Karnataka", "discom": "Gulbarga Electricity Supply Company Limited", "website": "gescom.in"},
            {"state": "Karnataka", "discom": "Bangalore Electricity Supply Company Limited", "website": "bescom.org"},
            {"state": "Karnataka", "discom": "Mangalore Electricity Supply Company Limited", "website": "mescom.in"},
            {"state": "Karnataka", "discom": "Hubli Electricity Supply Company Limited", "website": "hescom.co.in"},
            {"state": "Telangana", "discom": "Telangana State Southern Power Distribution Company Ltd", "website": "tssouthernpower.com"},
            {"state": "Telangana", "discom": "Telangana State Northern Power Distribution Company Ltd", "website": "tsnpdcl.in"},
            {"state": "Lakshadweep", "discom": "Electricity Department, UT of Lakshadweep", "website": "lakshadweep.gov.in"},
            {"state": "Puducherry", "discom": "Electricity Department, UT of Puducherry", "website": "electricity.py.gov.in"},
            {"state": "Andaman & Nicobar", "discom": "Electricity Department, UT of Andaman & Nicobar", "website": "electricity.andaman.gov.in"},
            {"state": "Andhra Pradesh", "discom": "Southern Power Distribution Company of A.P. Limited", "website": "apspdcl.in"},
            {"state": "Andhra Pradesh", "discom": "Andhra Pradesh Eastern Power Distribution Company Ltd", "website": "apepdcl.in"},
            {"state": "Tamil Nadu", "discom": "Tamil Nadu Generation & Distribution Corporation Limited", "website": "tangedco.gov.in"},

            # West Zone
            {"state": "Gujarat", "discom": "Uttar Gujarat Vij Company Limited", "website": "ugvcl.com"},
            {"state": "Gujarat", "discom": "Madhya Gujarat Vij Company Limited", "website": "mgvcl.com"},
            {"state": "Gujarat", "discom": "Paschim Gujarat Vij Company Limited", "website": "pgvcl.com"},
            {"state": "Gujarat", "discom": "Dakshin Gujarat Vij Company Limited", "website": "dgvcl.com"},
            {"state": "Goa", "discom": "Electricity Department, Government of Goa", "website": "electricity.goa.gov.in"},
            {"state": "Madhya Pradesh", "discom": "Madhya Pradesh Madhya Kshetra Vidyut Vitran Company Limited", "website": "mpcz.co.in"},
            {"state": "Madhya Pradesh", "discom": "MP Paschim Kshetra Vidyut Vitran Company Limited", "website": "mpwz.co.in"},
            {"state": "Madhya Pradesh", "discom": "MP Poorv Kshetra Vidyut Vitran Company Limited", "website": "mpez.co.in"},
            {"state": "Rajasthan", "discom": "Jaipur Vidyut Vitran Nigam Limited", "website": "energy.rajasthan.gov.in/jvvnl"},
            {"state": "Rajasthan", "discom": "Ajmer Vidyut Vitran Nigam Limited", "website": "energy.rajasthan.gov.in/avvnl"},
            {"state": "Rajasthan", "discom": "Jodhpur Vidyut vitran Nigam Limited", "website": "energy.rajasthan.gov.in/jdvvnl"},
            {"state": "Maharashtra", "discom": "Maharashtra State Electricity Distribution Co. Ltd.", "website": "mahadiscom.in"},
            {"state": "Maharashtra", "discom": "Brihmanmumbai Electric Supply Company", "website": "bestundertaking.com"},
            {"state": "Chhattisgarh", "discom": "Chhattisgarh State Power Distribution Company Ltd.", "website": "cspdcl.co.in"},
            {"state": "Dadra & Nagar Haveli", "discom": "Dadra & Nagar Haveli Power Distribution Corporation Ltd", "website": "electricity.dnh.gov.in"},
            {"state": "Daman & Diu", "discom": "Electricity Department, UT of Daman & Diu", "website": "electricity.daman.gov.in"},

            # East Zone
            {"state": "West Bengal", "discom": "West Bengal State Electricity Distribution Company Limited", "website": "wbsedcl.in"},
            {"state": "West Bengal", "discom": "Durgapur Project Limited", "website": "dpl.net.in"},
            {"state": "Jharkhand", "discom": "Jharkhand Bijli Vitran Nigam Limited", "website": "jbvnl.co.in"},
            {"state": "Odisha", "discom": "North Eastern Supply Company Limited", "website": "nescoodisha.com"},
            {"state": "Odisha", "discom": "Southern Electricity Supply Company Limited", "website": "southcoodisha.com"},
            {"state": "Odisha", "discom": "Central Electricity Supply Company Limited", "website": "cesuodisha.com"},
            {"state": "Odisha", "discom": "Western Electricity Supply Company of Odisha Limited", "website": "wescoodisha.com"}
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
