import hashlib
import numpy as np
import trimesh
import os
from PIL import Image

def create_7_sided_prism(height=1.0, radius=0.5, image_path=None):
    """
    Create a 7-sided prism with the same image on each side face, with no bottom face.
    """
    # Create vertices for the 7-sided base
    n_sides = 7
    angles = [-0.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5]
    angles = [angle * np.pi/15 for angle in angles]

    # Base vertices (bottom)
    base_vertices = np.column_stack((
        radius * np.cos(angles),
        np.zeros(n_sides),
        radius * np.sin(angles)
    ))
    base_vertices[:, 1] = -height / 5 * 2

    # Top vertices
    top_vertices = base_vertices.copy()
    top_vertices[:, 1] = height / 5 * 3
    
    # Combine all vertices
    vertices = np.vstack((base_vertices, top_vertices))
    # print(vertices)
    
    # Create faces (all must be triangles for trimesh)
    faces = []
    
    # Side faces (quads converted to triangles)
    for i in range(n_sides):
        i_next = (i + 1) % n_sides
        
        # Define the quad vertices (bottom-left, bottom-right, top-right, top-left)
        # Convert quad to two triangles
        faces.append([i, i_next, i + n_sides])
        faces.append([i_next, i_next + n_sides, i + n_sides])
    
    # No bottom face - leaving it open
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Apply the image texture if provided
    if image_path and os.path.exists(image_path):
        # Load the image
        image = Image.open(image_path)
        
        # Create a material with the image
        material = trimesh.visual.material.SimpleMaterial(image=image)
        
        # Create UV coordinates for all vertices
        new_vertices = []
        new_faces = []
        new_uvs = []
        
        # Process side faces
        for i in range(n_sides):
            i_next = (i + 1) % n_sides
            
            # Add 4 vertices for this quad (even if duplicates)
            v_bottom_left = vertices[i]
            v_bottom_right = vertices[i_next]
            v_top_right = vertices[i_next + n_sides]
            v_top_left = vertices[i + n_sides]
            
            # Add vertices
            idx_bl = len(new_vertices)
            new_vertices.append(v_bottom_left)
            idx_br = len(new_vertices)
            new_vertices.append(v_bottom_right)
            idx_tr = len(new_vertices)
            new_vertices.append(v_top_right)
            idx_tl = len(new_vertices)
            new_vertices.append(v_top_left)
            
            # Add UVs - full texture on each side
            new_uvs.append([0.0, 0.0])  # Bottom left
            new_uvs.append([1.0, 0.0])  # Bottom right
            new_uvs.append([1.0, 1.0])  # Top right
            new_uvs.append([0.0, 1.0])  # Top left
            
            # Add faces (two triangles)
            new_faces.append([idx_bl, idx_br, idx_tl])
            new_faces.append([idx_br, idx_tr, idx_tl])
        
        # Create new mesh with proper UV mapping
        new_vertices = np.array(new_vertices)
        new_faces = np.array(new_faces)
        new_uvs = np.array(new_uvs)
        
        mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            visual=trimesh.visual.TextureVisuals(
                uv=new_uvs,
                material=material
            )
        )
    
    return mesh

def main():
    # Path to the image to apply to each face
    prompt_file_name = "/workspace/prompts.txt"
    with open(prompt_file_name, "r") as file:
        cnt = 0
        for line in file:
            cnt += 1
            if cnt % 1000 == 0:
                print(f"--------------------{cnt}-------------------")
            prompt = line.strip()
            folder_id = hashlib.sha256(prompt.encode()).hexdigest()
            # print(folder_id)
            DB_PATH = "/workspace/FuckDB"
            image_path = f"{DB_PATH}/{folder_id}/img.jpeg"  # You need to provide this image
    
            # Create a default image if none is provided
            if not os.path.exists(image_path):
                print(f"Image not found at {image_path}, creating a default texture...")
                default_img = Image.new('RGB', (512, 512), color=(255, 200, 200))
                
                # Add some pattern to make it more visible
                from PIL import ImageDraw
                draw = ImageDraw.Draw(default_img)
                draw.rectangle([(100, 100), (400, 400)], outline=(0, 0, 0), width=5)
                draw.text((200, 250), "Side", fill=(0, 0, 0))
                
                image_path = "default_texture.jpg"
                default_img.save(image_path)
    
            # Create the prism
            prism = create_7_sided_prism(height=0.5, radius=0.55, image_path=image_path)
            
            # Export as GLB
            output_path = "7_sided_prism.glb"
            
            
            # Make sure directories exist before exporting
            # os.makedirs(os.path.dirname(f"/workspace/{id}/"), exist_ok=True)
            os.makedirs(os.path.dirname(f"/workspace/FuckDB/{folder_id}/"), exist_ok=True)
            
            # output_path1 = "../e48ee4473fe0684d1bd9c7614a2d24686c1dab3dfa7c66cf43a64cb5106522e3/mesh.glb"
            # output_path2 = "../PremiumDB/e48ee4473fe0684d1bd9c7614a2d24686c1dab3dfa7c66cf43a64cb5106522e3/mesh.glb"
            
            output_path2 = f"/workspace/FuckDB/{folder_id}/mesh.glb"
            prism.export(output_path)            
            prism.export(output_path2)
            # print(f"Prism exported to {output_path}")

if __name__ == "__main__":
    main()
