import sys
import boto3
import ast
import json
import nbformat
import zipfile
import tempfile
import fiona
import logging
import os
import rasterio

import re
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sanitize_tag_value(value: str) -> str:
    """Sanitize metadata values for Minio tags."""
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=._:/ "
    sanitized = "".join([c if c in allowed_chars else "_" for c in str(value)])
    return sanitized[:128]

def extract_python_metadata(code: str) -> List[Dict]:
    """Extract functions and classes from Python code using AST."""
    metadata = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_metadata = {
                    "type": "function",
                    "name": node.name,
                    "params": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node) or ""
                }
                metadata.append(func_metadata)
            elif isinstance(node, ast.ClassDef):
                class_metadata = {
                    "type": "class",
                    "name": node.name,
                    "methods": [],
                    "docstring": ast.get_docstring(node) or ""
                }
                metadata.append(class_metadata)
    except Exception as e:
        logging.warning(f"Python parsing error: {e}")
    return metadata

def extract_java_metadata(code: str) -> List[Dict]:
    """Extract methods and classes from Java code using regex."""
    metadata = []
    try:
        class_pattern = r'class\s+(\w+)\s*{([^}]*)}'
        method_pattern = r'(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+)\s*\(([^)]*)\)'
        
        for class_match in re.finditer(class_pattern, code, re.DOTALL):
            class_name = class_match.group(1)
            class_metadata = {
                "type": "class",
                "name": class_name,
                "methods": []
            }
            class_body = class_match.group(2)
            
            for method_match in re.finditer(method_pattern, class_body):
                return_type = method_match.group(0).split()[1]
                method_name = method_match.group(2)
                params = method_match.group(3).split(',') if method_match.group(3) else []
                
                method_metadata = {
                    "name": method_name,
                    "return_type": return_type,
                    "params": params
                }
                class_metadata["methods"].append(method_metadata)
            
            metadata.append(class_metadata)
    except Exception as e:
        logging.warning(f"Java parsing error: {e}")
    return metadata

def extract_notebook_metadata(notebook_path: str) -> List[Dict]:
    """Extract code cells and functions from Jupyter notebooks."""
    metadata = []
    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell_metadata = {
                    "type": "code_cell",
                    "source": cell.source.split('\n'),
                    "functions": extract_python_metadata(cell.source)
                }
                metadata.append(cell_metadata)
    except Exception as e:
        logging.warning(f"Notebook parsing error: {e}")
    return metadata

def process_zip(zip_path: str, bucket: str) -> Dict:
    """Process ZIP files and their contents based on bucket type."""
    metadata = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, tmpdir)
                        
                        try:
                            if bucket == "datasets":
                                if file.lower().endswith(('.tif', '.tiff')):
                                    with rasterio.open(file_path) as src:
                                        crs = src.crs.to_string() if src.crs else "undefined"
                                        metadata[file] = {
                                            "type": "raster",
                                            "crs": crs,
                                            "bounds": list(src.bounds),
                                            "resolution": [float(res) for res in src.res]
                                        }
                                elif file.lower().endswith(('.shp', '.geojson')):
                                    with fiona.open(file_path) as src:
                                        crs = dict(src.crs) if src.crs else {}
                                        metadata[file] = {
                                            "type": "vector",
                                            "crs": crs,
                                            "schema": src.schema,
                                            "bounds": list(src.bounds)
                                        }
                            
                            elif bucket == "code":
                                if file.endswith('.py'):
                                        with open(file_path, 'r') as f:
                                            content = f.read()
                                        metadata[rel_path] = {
                                            "type": "python",
                                            "analysis": extract_python_metadata(content)
                                        }
                                elif file.endswith('.java'):
                                    with open(file_path, 'r') as f:
                                        content = f.read()
                                    metadata[rel_path] = {
                                        "type": "java", 
                                        "analysis": extract_java_metadata(content)
                                    }
                        
                            elif bucket == "notebooks":
                                if file.endswith('.ipynb'):
                                        metadata[rel_path] = {
                                            "type": "notebook",
                                            "analysis": extract_notebook_metadata(file_path)
                                        }
                                
                        except Exception as e:
                            logging.warning(f"Failed to process {file}: {e}")
        except Exception as e:
            logging.error(f"ZIP processing failed: {e}")
    
    return metadata

def extract_metadata(bucket: str, key: str):
    tmp_path = None
    try:
        # Initialize S3 client for Minio
        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://i-guide-storage-dev.cis220065.projects.jetstream-cloud.org:9010"),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY_ID', ''),
            aws_secret_access_key=os.getenv('MINIO_ACCESS_KEY', ''),
            config=boto3.session.Config(signature_version='s3v4')
        )

        # Download file to temporary storage
        # The object key is caller-controlled. os.path.join("/tmp", key) followed it out of /tmp:
        # "../../etc/x" resolves above it, and an absolute key discards "/tmp" entirely, while
        # makedirs then created whatever directories the key named. Keep only the final name.
        safe_name = os.path.basename(key) or "downloaded-object"
        tmp_path = os.path.join("/tmp", safe_name)
        logging.info(f"Downloading {key} from bucket {bucket}")
        s3.download_file(bucket, key, tmp_path)
        logging.info(f"Download complete: {tmp_path}")

        # Process ZIP files
        metadata = {}
        if key.lower().endswith('.zip'):
            logging.info("Processing ZIP archive")
            metadata = process_zip(tmp_path, bucket)
        else:
            logging.warning("Non-ZIP file in code/notebooks bucket")
            return

        if not metadata:
            logging.warning("No metadata extracted")
            return

        logging.info(f"Extracted metadata: {json.dumps(metadata, indent=2)}")

        # Sanitize and format for tagging
        sanitized_metadata = {}
        for file_path, analysis in metadata.items():
            safe_prefix = sanitize_tag_value(os.path.splitext(file_path)[0]) + "_"
            
            if bucket == "code":
                file_analysis = analysis.get("analysis", []) if isinstance(analysis, dict) else []
                for i, item in enumerate(file_analysis):
                    sanitized_metadata[f"{safe_prefix}{i}_type"] = sanitize_tag_value(item["type"])
                    sanitized_metadata[f"{safe_prefix}{i}_name"] = sanitize_tag_value(item["name"])
                    
            elif bucket == "notebooks":
                notebook_cells = analysis.get("analysis", []) if isinstance(analysis, dict) else []
                for i, cell in enumerate(notebook_cells):
                    sanitized_metadata[f"{safe_prefix}cell_{i}_type"] = "code"
                    #sanitized_metadata[f"{safe_prefix}cell_{i}_funcs"] = str(len(cell["functions"]))

        # Attach metadata to Minio object
        """s3.put_object_tagging(
            Bucket=bucket,
            Key=key,
            Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in sanitized_metadata.items()]}
        )"""
        logging.info("Metadata extracted successfully")

    except NoCredentialsError:
        logging.error("Missing Minio credentials.")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        logging.error("Usage: python extract_metadata.py <bucket> <key>")
        sys.exit(1)
    bucket, key = sys.argv[1], sys.argv[2]
    extract_metadata(bucket, key)
