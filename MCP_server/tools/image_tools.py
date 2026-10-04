import base64
import requests
import os
import time
from server import mcp_tool
from fastapi import UploadFile, File

API_URL = os.environ.get("VISION_API_URL")
API_KEY = os.environ.get("VISION_API_KEY")
MODEL_NAME = os.environ.get(
    "VISION_API_MODEL",
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

# --- Helper ---
def encode_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

# --- MCP Tools ---
@mcp_tool(category="generation")
def describe_image(
    file: UploadFile = File(...),
    prompt_text: str = "Describe what is in this image."
) -> str:
    """Describe the contents of an image using the configured vision model.
    
    Args:
        file: The image file to analyze (uploaded file)
        prompt_text: Custom prompt for the vision model (default: "Describe what is in this image.")
    
    Returns:
        Text description of the image contents
    """
    image_bytes = file.file.read()
    file.file.close()
    image_b64 = encode_bytes(image_bytes)
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "stream": False
    }
    try:
        print(
            "describe_image: file_bytes=%s prompt_len=%s"
            % (len(image_bytes), len(prompt_text))
        )
        start_time = time.monotonic()
        print("describe_image: sending request...")
        response = requests.post(API_URL, headers=HEADERS, json=body, timeout=30)
        elapsed = time.monotonic() - start_time
        print("describe_image: got response status=%s in %.2fs" % (response.status_code, elapsed))
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Failed to describe image: {str(e)}")

@mcp_tool(
    category="generation",
    tool_description=(
        "Describe a map image with focus on area, problem, and provided information.\n\n"
        "Example usage:\n"
        'curl -X POST "http://localhost:8000/api/tool/describe_map" '
        '-F "file=@tgis_a_2343063_f0007_c.jpg"'
    )
)
def describe_map(
    file: UploadFile = File(...),
    prompt_text: str = (
        "Describe the given map. Focus on which area the map depicts, "
        "what problem the map describes, and what information the map provides. "
        "Format the response in markdown format."
    )
) -> str:
    """Describe a map image with focus on area, problem, and provided information.
    
    Args:
        file: The map image file to analyze (uploaded file)
        prompt_text: Custom prompt focusing on area, problem, and information (default provided)
    
    Returns:
        Markdown-formatted description of the map
    """
    image_bytes = file.file.read()
    file.file.close()
    image_b64 = encode_bytes(image_bytes)
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "stream": False
    }
    try:
        print(
            "describe_map: file_bytes=%s prompt_len=%s"
            % (len(image_bytes), len(prompt_text))
        )
        start_time = time.monotonic()
        print("describe_map: sending request...")
        response = requests.post(API_URL, headers=HEADERS, json=body)
        elapsed = time.monotonic() - start_time
        print("describe_map: got response status=%s in %.2fs" % (response.status_code, elapsed))
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Failed to describe image: {str(e)}")
