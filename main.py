import base64
import re
from json import JSONDecodeError
from typing import Any, Dict, List
from flask import Flask, json, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

genai.configure(api_key="GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')
MODEL_PLANT = tf.lite.Interpreter(model_path="models/model_int8_optimized.tflite")
MODEL_PLANT.allocate_tensors()

MODEL_ANIMAL = tf.lite.Interpreter(model_path="models/livestock_optimized.tflite")
MODEL_ANIMAL.allocate_tensors()
# Get input and output tensors
input_details_plant = MODEL_PLANT.get_input_details()
output_details_plant = MODEL_PLANT.get_output_details()

input_details_animal = MODEL_ANIMAL.get_input_details()
output_details_animal = MODEL_ANIMAL.get_output_details()

CLASS_NAMES_PLANT = [
    'Apple scab disease', 'Apple black rot disease', 'Apple cedar apple rust disease',
    'Apple healthy', 'Blueberry healthy',
    'Cherry (including sour) powdery mildew', 'Cherry (including sour) healthy',
    'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn (maize) Common rust',
    'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 'Grape Black rot',
    'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape healthy',
    'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot', 'Peach healthy',
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight',
    'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew',
    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
    'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot',
    'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato Mosaic Virus', 'Tomato healthy'
]

CLASS_NAMES_ANIMAL = ['Foot and mouth disease pigs',
 'Glassers disease pigs',
 'Healthy cow',
 'Healthy pigs',
 'Lumpy skin cow',
 'Pdns pigs'
]

livestock_names = [
    'cow',
    'cattle',
    'pigs',
    
]

plant_names = [
    'apple',
    'blueberry',
    'cherry',
    'corn',
    'grape',
    'orange',
    'peach',
    'pepper',
    'potato',
    'raspberry',
    'soybean',
    'squash',
    'strawberry',
    'tomato',
    'maize',
    'sugarcane',
    'cotton',
    'coffee',
    'tea',
    'tobacco',
    'wheat',
    'barley',
    'rice',
    
]

MAX_SIZE = 720

# Load your TFLite models

# Your CLASS_NAMES_PLANT, CLASS_NAMES_ANIMAL, livestock_names, and plant_names lists here
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and preprocess the image for the model."""
    image = image.resize((224, 224))  # Resize to 224x224
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array

def resize_image(image: Image.Image) -> Image.Image:
    """Resize image maintaining aspect ratio so that the largest dimension is MAX_SIZE."""
    try:
        if max(image.size) > MAX_SIZE:
            ratio = MAX_SIZE / max(image.size)
            new_size = tuple([int(x * ratio) for x in image.size])
            return image.resize(new_size, Image.LANCZOS)
        else:
            return image
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise ValueError(f"Unable to process image: {str(e)}")

def generate_symptoms(disease: str, category: str) -> List[str]:
    prompt = f"""
    List the top 5 symptoms of {disease} in {category}. 
    Provide the answer in JSON format with a key 'symptoms' and an array of exactly 5 strings.
    Example format: {{"symptoms": ["symptom1", "symptom2", "symptom3", "symptom4", "symptom5"]}}
    """
    response = model.generate_content(prompt)
    try:
        symptoms_json = json.loads(response.text)
        symptoms = symptoms_json.get('symptoms', [])
        if len(symptoms) != 5:
            raise ValueError("Incorrect number of symptoms returned")
        return symptoms
    except JSONDecodeError:
        raise ValueError("Invalid JSON response from Gemini")
    except KeyError:
        raise ValueError("Unexpected JSON structure in Gemini response")


def read_image_file(file: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file))

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    image = image.resize((224, 224))  # Resize the image to the expected input shape
    image = np.array(image) / 255.0
    return image

  # Normalize the image
def generate_treatment(symptoms: List[str],catagory: str) -> str:
    symptoms_str = ", ".join(symptoms)
    prompt = f"Given the following symptoms: {symptoms_str}, of {catagory} disease suggest a general treatment plan. Provide a concise response."
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_causes(disease: str, category: str, symptoms: List[str]) -> List[str]:
    prompt = f"""
    Given the disease '{disease}' in {category} with the following symptoms: {', '.join(symptoms)},
    list the top 3 potential causes.
    Provide your answer as a JSON array of exactly 3 strings, each representing a cause.
    Example: ["Cause 1", "Cause 2", "Cause 3"]
    Only provide the JSON array, no other text.
    """
    try:
        response = model.generate_content(prompt)
        
        # Check if the response was blocked
        if not response.candidates or response.candidates[0].finish_reason == "SAFETY":
            logger.warning("Gemini response was blocked due to safety concerns.")
            for candidate in response.candidates:
                for rating in candidate.safety_ratings:
                    logger.warning(f"Safety rating: {rating.category} - {rating.probability}")
            return ["Unable to generate causes due to content restrictions", 
                    "Please consult a professional for accurate information", 
                    "Safety measures prevented detailed response"]

        # If we have a valid response, process it
        if response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
            json_str = response_text.strip()
            if json_str.startswith('```json'):
                json_str = json_str.split('```json')[1]
            if json_str.endswith('```'):
                json_str = json_str.rsplit('```', 1)[0]
            
            result = json.loads(json_str)
            
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result[:3]  # Return up to 3 causes
            else:
                logger.warning(f"Unexpected response format: {result}")
                return ["Unexpected response format", 
                        "Please try again or consult a professional", 
                        "Unable to process the generated causes"]
        else:
            logger.warning("Empty response content from Gemini")
            return ["No causes generated", 
                    "Please try again or consult a professional", 
                    "Unable to process the request at this time"]
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response from Gemini: {response_text}")
        return ["Invalid response format", 
                "Please try again or consult a professional", 
                "Unable to process the generated causes"]
    except Exception as e:
        logger.error(f"Error in generate_causes: {str(e)}")
        logger.error(f"Full response: {response}")
        return ["Error generating causes", 
                "Please try again or consult a professional", 
                f"Technical details: {str(e)}"]

def generate_treatment_plan(disease: str, category: str, symptoms: List[str]) -> Dict[str, List[str]]:
    prompt = f"""
    Provide a treatment plan for {disease} in {category} with the following symptoms: {', '.join(symptoms)}.
    Structure your answer as a JSON object with the following keys:
    1. "immediate_actions": List of 2-3 immediate steps to take
    2. "long_term_management": List of 2-3 long-term management strategies
    3. "preventive_measures": List of 2-3 preventive measures
    Example:
    {{
        "immediate_actions": ["Action 1", "Action 2", "Action 3"],
        "long_term_management": ["Strategy 1", "Strategy 2", "Strategy 3"],
        "preventive_measures": ["Measure 1", "Measure 2", "Measure 3"]
    }}
    Only provide the JSON object, no other text.
    """
    response = model.generate_content(prompt)
    return process_gemini_response(response.text)

def generate_causes(disease: str, category: str, symptoms: List[str]) -> List[str]:
    prompt = f"""
    Given the disease '{disease}' in {category} with the following symptoms: {', '.join(symptoms)},
    list the top 3 potential causes.
    Provide your answer as a JSON array of exactly 3 strings, each representing a cause.
    Example: ["Cause 1", "Cause 2", "Cause 3"]
    Only provide the JSON array, no other text.
    """
    try:
        response = model.generate_content(prompt)
        
        # Check if the response was blocked
        if not response.candidates or response.candidates[0].finish_reason == "SAFETY":
            logger.warning("Gemini response was blocked due to safety concerns.")
            for candidate in response.candidates:
                for rating in candidate.safety_ratings:
                    logger.warning(f"Safety rating: {rating.category} - {rating.probability}")
            return ["Unable to generate causes due to content restrictions", 
                    "Please consult a professional for accurate information", 
                    "Safety measures prevented detailed response"]

        # If we have a valid response, process it
        if response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
            json_str = response_text.strip()
            if json_str.startswith('```json'):
                json_str = json_str.split('```json')[1]
            if json_str.endswith('```'):
                json_str = json_str.rsplit('```', 1)[0]
            
            result = json.loads(json_str)
            
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result[:3]  # Return up to 3 causes
            else:
                logger.warning(f"Unexpected response format: {result}")
                return ["Unexpected response format", 
                        "Please try again or consult a professional", 
                        "Unable to process the generated causes"]
        else:
            logger.warning("Empty response content from Gemini")
            return ["No causes generated", 
                    "Please try again or consult a professional", 
                    "Unable to process the request at this time"]
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response from Gemini: {response_text}")
        return ["Invalid response format", 
                "Please try again or consult a professional", 
                "Unable to process the generated causes"]
    except Exception as e:
        logger.error(f"Error in generate_causes: {str(e)}")
        logger.error(f"Full response: {response}")
        return ["Error generating causes", 
                "Please try again or consult a professional", 
                f"Technical details: {str(e)}"]

def parse_treatment(treatment_text: str) -> Dict[str, List[str]]:
    # Remove the "General Treatment Plan:" header
    treatment_text = re.sub(r'^General Treatment Plan:\s*', '', treatment_text, flags=re.IGNORECASE)
    
    # Split the text into sections
    sections = re.split(r'\n\s*-\s*', treatment_text)
    
    parsed_treatment = {}
    for section in sections:
        if section.strip():
            # Split each section into title and items
            title, *items = re.split(r':\s*\n', section)
            title = title.strip()
            if items:
                # Remove bullet points and create a list of items
                items = [item.strip('- ') for item in items[0].split('\n') if item.strip()]
                parsed_treatment[title] = items
            else:
                parsed_treatment[title] = []
    
    return parsed_treatment

def process_gemini_response(response_text: str) -> Any:
    json_str = response_text.strip()
    if json_str.startswith('```json'):
        json_str = json_str.split('```json')[1]
    if json_str.endswith('```'):
        json_str = json_str.rsplit('```', 1)[0]
    
    try:
        return json.loads(json_str)
    except JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        logger.error(f"Problematic JSON string: {json_str}")
        return None  # or return a default value

def analyze_image_for_disease(image_data: bytes, name: str, symptoms: List[str]) -> Dict[str, Any]:
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Prepare the prompt
    prompt = f"""
    Analyze this image of a {name} and determine the most likely disease.
    Provide your answer in JSON format with two keys: 'disease' (a string with the name of the disease) 
    and 'confidence' (a float between 0 and 1 representing your confidence in the diagnosis).
    Example: {{"disease": "Apple scab", "confidence": 0.85}}
    Only provide the JSON object, no other text.
    """

    try:
        # Generate content with Gemini
        initial_response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_str}])
        result = process_gemini_response(initial_response.text)

        # Generate other possible symptoms
        other_symptoms_prompt = f"""
        Given the disease '{result['disease']}' in {name} with the following symptoms: {', '.join(symptoms)},
        list 3 other possible symptoms that might occur but are not mentioned.
        Provide the answer in JSON format with a key 'other_symptoms' and an array of exactly 3 strings.
        Example format: {{"other_symptoms": ["symptom1", "symptom2", "symptom3"]}}
        """
        other_symptoms_response = model.generate_content(other_symptoms_prompt)
        other_symptoms = process_gemini_response(other_symptoms_response.text)

        # Generate causes
        causes = generate_causes(result['disease'], name, symptoms)

        # Generate treatment plan
        treatment_plan = generate_treatment_plan(result['disease'], name, symptoms)

        return {
            "name": name,
            "disease": result['disease'],
            "confidence": result['confidence'],
            "symptoms": symptoms,
            "other_possible_symptoms": other_symptoms['other_symptoms'] if isinstance(other_symptoms, dict) else [],
            "potential_causes": causes,
            "treatment_plan": treatment_plan
        }
    except Exception as e:
        logger.error(f"Error in analyze_image_for_disease: {str(e)}", exc_info=True)
        raise ValueError(f"Error analyzing image: {str(e)}")

def process_gemini_response(response_text: str) -> Any:
    json_str = response_text.strip()
    if json_str.startswith('```json'):
        json_str = json_str.split('```json')[1]
    if json_str.endswith('```'):
        json_str = json_str.rsplit('```', 1)[0]
    
    try:
        return json.loads(json_str)
    except JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        logger.error(f"Problematic JSON string: {json_str}")
        return None  # or return a default value

def generate_top_symptoms(disease: str, category: str, given_symptoms: List[str]) -> List[str]:
    prompt = f"""
    Given the disease '{disease}' in {category} and the following symptoms: {', '.join(given_symptoms)},
    list the top 5 most common symptoms for this disease, including those already mentioned if applicable.
    Provide ONLY the JSON response with a key 'symptoms' and an array of exactly 5 strings.
    Example format: {{"symptoms": ["symptom1", "symptom2", "symptom3", "symptom4", "symptom5"]}}
    """
    response = model.generate_content(prompt)
    try:
        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            symptoms_json = json.loads(json_str)
            symptoms = symptoms_json.get('symptoms', [])
            if len(symptoms) != 5:
                raise ValueError("Incorrect number of symptoms returned")
            return symptoms
        else:
            raise ValueError("No JSON found in the response")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response from Gemini: {response.text}")
        raise ValueError("Invalid JSON response from Gemini")
    except KeyError:
        logger.error(f"Unexpected JSON structure in Gemini response: {response.text}")
        raise ValueError("Unexpected JSON structure in Gemini response")
    except Exception as e:
        logger.error(f"Error in generate_top_symptoms: {str(e)}", exc_info=True)
        raise ValueError(f"Error generating top symptoms: {str(e)}")
    
def generate_disease_info(disease: str, category: str) -> Dict[str, Any]:
    try:
        # Generate diagnosis
        diagnosis_prompt = f"Provide a brief diagnosis for {disease} in {category}. Keep it concise, about 2-3 sentences."
        diagnosis_response = model.generate_content(diagnosis_prompt)
        diagnosis = diagnosis_response.text.strip()

        # Generate possible treatment
        treatment_prompt = f"Suggest a possible treatment for {disease} in {category}. Provide 3-4 concise steps."
        treatment_response = model.generate_content(treatment_prompt)
        treatment = treatment_response.text.strip().split('\n')

        # Generate possible causes
        causes_prompt = f"""
        List the top 3 potential causes of {disease} in {category}.
        Provide your answer as a JSON array of exactly 3 strings, each representing a cause.
        Example: ["Cause 1", "Cause 2", "Cause 3"]
        Only provide the JSON array, no other text.
        """
        causes_response = model.generate_content(causes_prompt)
        causes = process_gemini_response(causes_response.text)

        # Generate symptoms
        symptoms_prompt = f"List the top 5 most common symptoms of {disease} in {category}. Provide your answer as a JSON array of exactly 5 strings."
        symptoms_response = model.generate_content(symptoms_prompt)
        symptoms = process_gemini_response(symptoms_response.text)

        return {
            "diagnosis": diagnosis,
            "possible_treatment": treatment,
            "possible_causes": causes,
            "symptoms": symptoms
        }
    except Exception as e:
        logger.error(f"Error in generate_disease_info: {str(e)}", exc_info=True)
        raise ValueError(f"Error generating disease information: {str(e)}")
    

def preprocess_image_tflite(image_data, input_shape):
    # Check if image_data is already a numpy array
    if isinstance(image_data, np.ndarray):
        # If it's a 1D array (raw bytes), convert it to an image
        if image_data.ndim == 1:
            img = Image.open(io.BytesIO(image_data))
        else:
            # If it's a 2D or 3D array, assume it's already an image
            img = Image.fromarray(image_data)
    else:
        # If it's bytes data, open it as an image
        img = Image.open(io.BytesIO(image_data))
    
    # Resize the image
    img = img.resize((input_shape[1], input_shape[2]))
    
    # Convert to RGB if it's not already
    img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Scale to [0, 255] and convert to INT8
    img_array = (img_array * 255.0).astype(np.int8)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_with_tflite(interpreter, input_details, output_details, image_array):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Hello, I am alive"})

# Convert your FastAPI functions to Flask routes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        name = request.form.get('name', '')
        symptoms = request.form.getlist('symptoms[]',)
        
        # Open the image file
        image = Image.open(file.stream)
        
        # Log the original image size
        logger.info(f"Original image size: {image.size}")
        
        # Resize the image
        resized_image = resize_image(image)
        
        # Log the resized image size
        logger.info(f"Resized image size: {resized_image.size}")
        
        # Convert the image to bytes for further processing if needed
        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        name_lower = name.lower()
        if name_lower in livestock_names:
            logger.info(f"Using livestock model for {name}")
            input_shape = input_details_animal[0]['shape']
            test_image = preprocess_image_tflite(img_byte_arr, input_shape)
            predictions = predict_with_tflite(MODEL_ANIMAL, input_details_animal, output_details_animal, test_image)
            CLASS_NAMES = CLASS_NAMES_ANIMAL
        elif name_lower in plant_names:
            logger.info(f"Using plant model for {name}")
            input_shape = input_details_plant[0]['shape']
            test_image = preprocess_image_tflite(img_byte_arr, input_shape)
            predictions = predict_with_tflite(MODEL_PLANT, input_details_plant, output_details_plant, test_image)
            CLASS_NAMES = CLASS_NAMES_PLANT
        else:
            logger.info(f"Using Gemini for prediction of {name}")
            result = analyze_image_for_disease(img_byte_arr, name_lower)
            logger.info(f"Gemini analysis result: {result}")
            return jsonify(result)

        predicted_class = np.argmax(predictions[0])
        class_name = CLASS_NAMES[predicted_class]
        confidence = float(predictions[0][predicted_class])
        confidence = min(confidence / 10 , 1.0)

        logger.info(f"Prediction: {class_name}, Confidence: {confidence}, symptoms: {symptoms}")


        if confidence < 0.70:
            logger.info(f"Low confidence ({confidence}), using Gemini as fallback")
            result = analyze_image_for_disease(img_byte_arr, name_lower, symptoms)
            logger.info(f"Gemini fallback result: {result}")
            return jsonify(result)
        else:
            logger.info(f"High confidence ({confidence}), using TFLite model for {class_name}")
            result = {
                "disease": class_name,
                "confidence": confidence
            }
        
        logger.info(f"Prediction: {class_name}, Confidence: {confidence}")
        
        other_symptoms = generate_top_symptoms(result["disease"], name, symptoms)
        causes = generate_causes(result["disease"], name, symptoms)
        treatment_plan = generate_treatment_plan(result["disease"], name, symptoms)
        
        final_result = {
            "name": name,
            "disease": result.get("disease", "Unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "given_symptoms": symptoms,
            "top_symptoms": other_symptoms,
            "potential_causes": causes,
            "treatment_plan": treatment_plan
        }
        
        logger.info(f"Final result: {final_result}")
        return jsonify(final_result)    

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    


@app.route("/predict_disease_from_symptoms", methods=['POST'])
def predict_disease_from_symptoms():
    try:
        # Try to get JSON data
        data = request.get_json()
        
        # If JSON data is not available, try to get form data
        if data is None:
            data = request.form.to_dict()
        
        # Extract the required fields
        name = data.get('name', '')
        location = data.get('location', '')
        symptoms = data.get('symptoms', [])
        environment = data.get('environment', '')
        age = data.get('age', '')

        # Convert symptoms to list if it's a string
        if isinstance(symptoms, str):
            symptoms = symptoms.split(',')

        # Log the received data
        logger.info(f"Received data: {data}")

        # Check if essential data is present
        if not name or not symptoms:
            return jsonify({
                "error": "Missing required data. Please provide at least a name and symptoms."
            }), 400

        # Your existing code to generate the prompt and get the response
        prompt = f"""
        Given the following information about a {name}:
        - Location: {location}
        - Symptoms: {', '.join(symptoms)}
        - Environment: {environment}
        - Approximate age: {age}

        Predict the most likely disease and provide the following information:
        1. Disease name
        2. Confidence level (a number between 0 and 1, where 1 is highest confidence)
        3. Brief diagnosis (2-3 sentences)
        4. Possible treatment (3-4 steps)
        5. 3 potential causes

        Format your response as a JSON object with the following structure:
        {{
            "disease": "Predicted disease name",
            "confidence": 0.85,
            "diagnosis": "Brief diagnosis",
            "possible_treatment": ["Step 1", "Step 2", "Step 3", "Step 4"],
            "possible_causes": ["Cause 1", "Cause 2", "Cause 3"]
        }}
        Ensure your response is a valid JSON object and nothing else.
        """

        response = model.generate_content(prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in the response")

        # Ensure confidence is a float between 0 and 1
        if 'confidence' in result:
            result['confidence'] = float(result['confidence'])
            result['confidence'] = max(0, min(result['confidence'], 1))
        else:
            result['confidence'] = 0.0

        # Add the input information to the result
        result["name"] = name
        result["location"] = location
        result["symptoms"] = symptoms
        result["environment"] = environment
        result["age"] = age

        logger.info(f"Disease prediction result: {result}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in predict_disease: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error predicting disease: {str(e)}"}), 500

# Include all your helper functions here (preprocess_image_tflite, predict_with_tflite, analyze_image_for_disease, etc.)


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8000))
#     app.run(host='0.0.0.0', port=port)
