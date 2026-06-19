import os
import base64
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from vllm import LLM, SamplingParams
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# ----------- A100 40GB Optimized Config -----------
BATCH_SIZE = 128  
GPU_UTILIZATION = 0.95
TARGET_LONGEST_DIM = 1288 

# ----------- Model Initialization -----------
print("Initializing vLLM on A100 40GB...")
llm = LLM(
    model="allenai/olmOCR-2-7B-1025",
    trust_remote_code=True,
    max_model_len=8192,
    limit_mm_per_prompt={"image": 1},
    gpu_memory_utilization=GPU_UTILIZATION, # Slightly safer
    enable_chunked_prefill=True,  # Crucial for A100 throughput
    max_num_seqs=64,             # A100 can handle more concurrent sequences
)

sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=4096,
)

# --- Directories ---
base_input_dir = '/home/azureuser/export_asbi'
output_path_dir = 'vlmOutput'
os.makedirs(output_path_dir, exist_ok=True)

# --- Pre-build prompt ---
base_prompt = build_no_anchoring_v4_yaml_prompt()
custom_instruction = " Please indicate the end of the document by adding '---END_OF_DOCUMENT---' if you believe you have processed the last page."
full_prompt = base_prompt + custom_instruction

# --- Helper Functions ---
def encode_image_to_base64(image_path, target_dim):
    """Loads, resizes, and encodes image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles CMYK or Grayscale JPEGs)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            w, h = img.size
            scale = target_dim / max(w, h)
            if scale < 1.0:

                new_size = (int(w * scale), int(h * scale))

                img = img.resize(new_size, Image.Resampling.LANCZOS)

            

            buffered = BytesIO()

            img.save(buffered, format="PNG")

            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:

        print(f"Error processing image {image_path}: {e}")

        return None



# --- Discovery & Filtering Phase ---

print(f"Scanning {base_input_dir} for image directories...")

all_leaf_folders = []

for root, dirs, files in os.walk(base_input_dir):

    # Only consider folders that contain JPEGs

    if any(f.lower().endswith(('.jpg', '.jpeg')) for f in files):

        all_leaf_folders.append(root)



folders_to_process = []

for folder in all_leaf_folders:

    # 1. Get the path relative to the base (e.g., '8720_0/1_0_0')

    rel_path = os.path.relpath(folder, base_input_dir)

    

    # 2. Flatten the path into a unique filename (e.g., '8720_0_1_0_0.txt')

    safe_filename = rel_path.replace(os.sep, '_') + ".txt"

    output_filepath = os.path.join(output_path_dir, safe_filename)

    

    # 3. Only add to queue if the output file DOES NOT exist

    if not os.path.exists(output_filepath):

        folders_to_process.append({

            'full_path': folder,

            'output_path': output_filepath,

            'display_name': rel_path

        })



print(f"Total image folders found: {len(all_leaf_folders)}")

print(f"Folders already processed: {len(all_leaf_folders) - len(folders_to_process)}")

print(f"New folders to process:    {len(folders_to_process)}")



# --- Configuration for Status Tracking ---
STATUS_FILE = "current_task.txt"

# --- Execution Phase ---

for task in tqdm(folders_to_process, desc="Overall Progress"):

    folder_path = task['full_path']

    output_filepath = task['output_path']

# 1. Update the status file with the current folder
    with open(STATUS_FILE, "w", encoding="utf-8") as sf:
        sf.write(f"PROCESSING: {task['display_name']}\nStarted at: {torch.cuda.Event(enable_timing=True)}")

    # Get sorted list of images to maintain document order

    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))])

    

    if not images:

        continue



    # Initialize output file immediately to mark folder as "in progress"

    with open(output_filepath, 'w', encoding='utf-8') as f:

        f.write(f"--- OCR START: {task['display_name']} ---\n")



    # Batch through the images in the folder

    for i in range(0, len(images), BATCH_SIZE):

        batch_files = images[i : i + BATCH_SIZE]

        batch_messages = []

        valid_filenames = []



        for img_name in batch_files:

            img_full_path = os.path.join(folder_path, img_name)

            img_b64 = encode_image_to_base64(img_full_path, TARGET_LONGEST_DIM)

            

            if img_b64:

                batch_messages.append([

                    {

                        "role": "user",

                        "content": [

                            {"type": "text", "text": full_prompt},

                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}

                        ],

                    }

                ])

                valid_filenames.append(img_name)



        # Run inference on the batch

        if batch_messages:

            try:

                outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)

                

                # Append results to the file

                with open(output_filepath, "a", encoding="utf-8") as f:

                    for idx, output in enumerate(outputs):

                        f.write(f"\n\n{'='*30} FILE: {valid_filenames[idx]} {'='*30}\n")

                        f.write(output.outputs[0].text)
                           
            except Exception as e:

                print(f"\nCritical error during VLLM inference in {folder_path}: {e}")



print("\nProcessing complete. All discovered archive folders have been processed.")