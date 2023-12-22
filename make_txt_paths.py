import glob

def save_file_names_to_text(folder_path, extension, output_file):
    # Find all files with the specified extension in the folder
    file_names = glob.glob(f"{folder_path}/*.{extension}")

    # Save file names to a text document
    with open(output_file, 'w') as file:
        for file_name in file_names:
            file.write(f"{file_name}\n")

# Example usage:
folder_path = "/home/UFAD/jfolden/datasets/nyudepthv2/val/official"  # Replace with the actual path to your folder
extension = "h5"
output_file = "Nyu_val.txt"

save_file_names_to_text(folder_path, extension, output_file)