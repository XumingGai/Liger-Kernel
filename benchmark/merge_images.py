from PIL import Image
import os
import math

def merge_images(image_folder, output_path, images_per_row=3, padding=10, background_color=(255, 255, 255)):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()

    images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

    if not images:
        print("No images found.")
        return

    width, height = images[0].size

    num_images = len(images)
    rows = math.ceil(num_images / images_per_row)

    merged_width = images_per_row * width + (images_per_row - 1) * padding
    merged_height = rows * height + (rows - 1) * padding

    merged_image = Image.new('RGB', (merged_width, merged_height), background_color)

    for index, image in enumerate(images):
        row = index // images_per_row
        col = index % images_per_row

        x = col * (width + padding)
        y = row * (height + padding)

        merged_image.paste(image, (x, y))

    merged_image.save(output_path)
    print(f"Merged image saved to: {output_path}")

if __name__ == "__main__":
    merge_images(
        image_folder="./test_folder",  
        output_path="merged_result.jpg",   
        images_per_row=3,                  
        padding=10                        
    )

