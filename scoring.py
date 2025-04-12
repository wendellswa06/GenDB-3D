import hashlib
import os
import time
import numpy as np
from fastapi import HTTPException
from models import ValidateRequest, ValidateResponse
from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel
from validation.text_similarity_model import TextSimilarityModel
from claude_integration import get_render_img_descs, get_prev_img_desc
from rendering import render, load_image
from image_insight import ImageAnalysisToolkit

DATA_DIR = '/workspace/FuckDB'
score_file_path = "/workspace/scores.txt"
EXTRA_PROMPT = 'anime'
FAILED_NAME = "/workspace/failed.txt"

text_model = TextModel()
image_model = ImageModel()
quality_model = QualityModel()
text_similarity_model = TextSimilarityModel()
image_vision_model = ImageAnalysisToolkit()

print("loading models")
    
text_model.load_model()
image_model.load_model()
quality_model.load_model()
text_similarity_model.load_model()
        
def validate(prompt:str):
    file = open(score_file_path, "a")
    print("----------------- Validation started -----------------")
    start = time.time()
    prompt = prompt
    # prompt = prompt.strip()
    file.write("----------------------------------------------------\n")
    file.write(f"{prompt}\n")
    # id = "0a0a5c75db9dbbffbd098321ef9a2b3bda0e97acdf016c280c7e6a06e02512d4"
    id = hashlib.sha256(prompt.encode()).hexdigest()
    print(prompt, id)
    print("Rendering 3D mesh file.....")    
    rendered_images, before_images, render_image_paths = render(prompt=prompt, id=id, verbose=False)
    
    image_descs = get_render_img_descs()
    print(image_descs)
    render_vectors = text_similarity_model.fetch_vectors(image_descs)
    
    prompt_vector = text_similarity_model.fetch_vectors([prompt])[0]
    
    prev_img_path = os.path.join(DATA_DIR, f"{id}/img.jpeg")

    image_paths = render_image_paths + [prev_img_path]

    is_like_real_object_image = image_vision_model.analyze_images(image_paths)
    file.write(f"is_like_real_object_image: {is_like_real_object_image}\n")
    print(f"Is real object image: {is_like_real_object_image}")

    if is_like_real_object_image == False:
        return 0

    print("Loading preview image.....")
    prev_img = load_image(prev_img_path)
    prev_img_desc = get_prev_img_desc(prev_img_path)
    prev_img_vector = text_similarity_model.fetch_vectors([prev_img_desc])
    
    Q0 = quality_model.compute_quality(prev_img_path)
    print(f"Q0: {Q0}")
    file.write(f"Q0: {Q0}\n")
    S0 = text_similarity_model.compute_semantic_similarity(prompt_vector, prev_img_vector)[0] if Q0 > 0.15 else 0
    print(f"S0: {S0} - taken time: {time.time() - start}")
    if S0 < 0.23:
        return 0
        
    Ri = detect_outliers([image_model.compute_clip_similarity(prev_img, img) for img in rendered_images])

    Si = detect_outliers(text_similarity_model.compute_semantic_similarity(prompt_vector, render_vectors))
    
    print(f"R0: taken time: {time.time() - start}")
    
    Qi = detect_outliers([quality_model.compute_quality(img) for img in before_images])
    
    S_geo = np.exp(np.log(Si).mean())
    R_geo = np.exp(np.log(Ri).mean())
    Q_geo = np.exp(np.log(Qi).mean())
    
    print("---- Rendered images similarities with preview image ---")
    print(Ri)
    print(f"R_geo: {R_geo}")
    
    print("---- Rendered images similarities with text prompt ----")
    print(Si)
    print(f"S_geo: {S_geo}")
    
    print("---- Rendered images quality ----")
    print(Qi)
    print(f"Q_geo: {Q_geo}")
    
    total_score = S0 * 0.25 + S_geo * 0.5 + R_geo * 0.3 + Q_geo * 0.1
    
    file.write(f"S0: {S0}\n")
    file.write(f"S_geo: {S_geo}\n")
    file.write(f"R_geo: {R_geo}\n")
    file.write(f"Q_geo: {Q_geo}\n")
    file.write(f"Total: {total_score}\n")
    file.write("----------------------------------------------------\n")
    print(f"---- Total Score: {total_score} ----")

    
    if total_score < 0.35:
        return 0
    return total_score

def real_object_check(prompt:str):
    prompt = prompt
    prompt = prompt.strip()
        
    id = hashlib.sha256(prompt.encode()).hexdigest()
    print(prompt, id)
    print("Rendering 3D mesh file.....")    
    rendered_images, before_images, render_image_paths = render(prompt=prompt, id=id, verbose=False)    
    
    prev_img_path = os.path.join(DATA_DIR, f"{id}/img.jpeg")

    image_paths = render_image_paths + [prev_img_path]

    is_like_real_object_image = image_vision_model.analyze_images(image_paths)
    

    if is_like_real_object_image == False:
        if os.path.exists(FAILED_NAME):
            outfile = open(FAILED_NAME, "a")
            outfile.write(f"{prompt}\n")
            outfile.close()
        
        return 0
    return 1

def detect_outliers(data, threshold=1.1):
    # Calculate Q1 and Q3
    sorted_data = sorted(data)
    Q1 = np.percentile(sorted_data, 25)
    Q3 = np.percentile(sorted_data, 75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Determine bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Identify non-outliers
    non_outliers = [x for x in data if lower_bound <= x <= upper_bound]
    
    return non_outliers

def quality_detect(prompt:str):
    
    start = time.time()
    prompt = prompt
    
    id = hashlib.sha256(prompt.encode()).hexdigest()
    # print(prompt, id)
    
    prev_img_path = os.path.join(DATA_DIR, f"{id}/img.jpeg")

    # print("Loading preview image.....")
    prev_img = load_image(prev_img_path)
    
    Q0 = quality_model.compute_quality(prev_img_path)
    # print(f"Q0: {Q0}")
    return Q0

    
if __name__ == "__main__":
    prompt_file_path = "/workspace/sample.txt"
    low_quality_path = "/workspace/low_quality_check.txt"
    lowfile = open(low_quality_path, "a")
    with open(prompt_file_path, "r") as file:
        cnt = 0
        for line in file:
            cnt += 1
            if cnt % 100 == 0:
                print(f"---------------------------------{cnt}---------------------------")
            prompt = line.strip()
            # is_like_object = real_object_check(prompt)
            Q0 = quality_detect(prompt)
            if Q0 < 0.15:
                lowfile.write(f"{prompt}\n")
    lowfile.close()


            
