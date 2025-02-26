import runpod
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
from openai import OpenAI
import os
import io
import re
import base64

# Initialize the OCR model with CUDA support
try:
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).cuda()
except Exception as e:
    print(f"Error initializing OCR model: {str(e)}")
    raise

# Initialize OpenAI client with better error handling
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    raise

def extract_text_from_pdf(pdf_file_bytes):
    try:
        doc = DocumentFile.from_pdf(pdf_file_bytes)
        result = model(doc)
        return result
    except Exception as e:
        print(f"Error in PDF extraction: {str(e)}")
        raise

def clean_extracted_text(result):
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join(word.value for word in line.words)
                extracted_text.append(line_text)
    return '\n'.join(extracted_text)

def extract_critical_information(text):
    try:
        MODEL = "gpt-4o"  # Using the original model
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert load confirmation analyst with extensive experience in the trucking and logistics industry. Your task is to extract critical information from load confirmation documents with perfect accuracy. Focus on identifying all key data points that carriers need for successful load execution and proper payment."},

                {"role": "user", "content": f"Given the raw information extracted using OCR from a load confirmation document, you should extract the most important parts such as rate information, pickup/delivery details, BOL numbers, reference numbers, and all other critical elements. The OCR results are stored in the variable `result`.\n\nOCR Result:\n{text}\n\nPlease provide the extracted information in JSON format.\n\nExample JSON output:\n{{\n  \"load_details\": {{\n    \"broker_name\": \"\",\n    \"broker_mc_number\": \"\",\n    \"load_confirmation_number\": \"\",\n    \"order_number\": \"\",\n    \"bol_number\": \"\",\n    \"reference_numbers\": [],\n    \"commodity\": \"\",\n    \"weight\": \"\",\n    \"piece_count\": \"\",\n    \"temperature_requirements\": \"\",\n    \"equipment_type\": \"\",\n    \"total_miles\": \"\"\n  }},\n  \"financial\": {{\n    \"base_rate\": \"\",\n    \"total_carrier_pay\": \"\",\n    \"accessorial_charges\": {{}},\n    \"detention_rate\": \"\",\n    \"detention_terms\": \"\",\n    \"payment_terms\": \"\"\n  }},\n  \"pickup\": {{\n    \"facility_name\": \"\",\n    \"address\": \"\",\n    \"city\": \"\",\n    \"state\": \"\",\n    \"zip\": \"\",\n    \"date\": \"\",\n    \"time_window\": \"\",\n    \"contact_information\": \"\",\n    \"reference_numbers\": [],\n    \"special_instructions\": \"\"\n  }},\n  \"delivery\": {{\n    \"facility_name\": \"\",\n    \"address\": \"\",\n    \"city\": \"\",\n    \"state\": \"\",\n    \"zip\": \"\",\n    \"date\": \"\",\n    \"time_window\": \"\",\n    \"contact_information\": \"\",\n    \"reference_numbers\": [],\n    \"special_instructions\": \"\"\n  }},\n  \"driver_equipment\": {{\n    \"driver_name\": \"\",\n    \"driver_phone\": \"\",\n    \"tractor_number\": \"\",\n    \"trailer_number\": \"\",\n    \"tractor_vin\": \"\"\n  }},\n  \"operational_requirements\": {{\n    \"tracking_requirements\": \"\",\n    \"communication_protocols\": \"\",\n    \"loading_responsibility\": \"\",\n    \"unloading_responsibility\": \"\"\n  }},\n  \"penalties_restrictions\": {{\n    \"cancellation_fee\": \"\",\n    \"rescheduling_fee\": \"\",\n    \"late_delivery_penalty\": \"\",\n    \"weekend_holiday_restrictions\": \"\"\n  }}\n}}\n\nExtracted Information JSON: Warning: Extract only information that is actually present in the document. Don't make up fake information. If certain fields are not present, leave them as empty strings."}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI extraction: {str(e)}")
        raise

def clean_and_convert_to_json(input_string):
    # Remove markdown code block indicators and whitespace
    cleaned_string = input_string.strip()
    cleaned_string = re.sub(r'^```json\s*|\s*```$', '', cleaned_string, flags=re.MULTILINE)
    
    # Remove non-printable characters except newlines
    cleaned_string = ''.join(char for char in cleaned_string if char.isprintable() or char in '\n\r')
    
    # Ensure proper JSON structure
    cleaned_string = cleaned_string.strip()
    if not cleaned_string.startswith('{'):
        cleaned_string = '{' + cleaned_string
    if not cleaned_string.endswith('}'):
        cleaned_string = cleaned_string + '}'
    
    try:
        data = json.loads(cleaned_string)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic string:\n{cleaned_string}")
        return {"error": f"JSON parsing error: {str(e)}"}

def handler(event):
    """
    RunPod handler function
    """
    try:
        # Get input data
        job_input = event["input"]
        
        if "base64_pdf" not in job_input:
            return {"error": "No PDF data provided"}
        
        # Decode base64 PDF
        try:
            pdf_bytes = base64.b64decode(job_input["base64_pdf"])
        except Exception as e:
            return {"error": f"Invalid base64 PDF data: {str(e)}"}
        
        # Process the PDF
        result = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        cleaned_text = clean_extracted_text(result)
        extracted_info = extract_critical_information(cleaned_text)
        json_result = clean_and_convert_to_json(extracted_info)
        
        return {
            "success": True,
            "data": json_result,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})