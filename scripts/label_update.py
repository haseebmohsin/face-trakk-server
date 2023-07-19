import os

def change_image_labels(folder_path, new_label):
    counter = 1
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):  # Add more file extensions if needed
            file_path = os.path.join(folder_path, file_name)
            
            # Get the file extension
            file_extension = os.path.splitext(file_name)[1]
            
            # Construct the new file name with the new label and counter
            new_file_name = f"{new_label}_{counter}{file_extension}"
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"Changed label of {file_name} to {new_file_name}")
            
            counter += 1


# Example usage
folder_path = "Cluster_3"
new_label = "XYZ"

change_image_labels(folder_path, new_label)
