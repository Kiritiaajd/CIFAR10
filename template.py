import os

# Define project structure
project_structure = {
    "CIFAR10-Image-Classification": {
        "data": ["train", "test", "README.md"],
        "models": ["cnn_model.py", "utils.py", "README.md"],
        "notebooks": ["cifar10_classification.ipynb", "README.md"],
        "scripts": ["train.py", "evaluate.py", "predict.py", "README.md"],
        "requirements.txt": None,
        "README.md": None,
        "LICENSE": None
    }
}

# Function to create project structure
def create_project_structure(base_dir, structure):
    for dir_name, content in structure.items():
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        # Handle file creation in the directories
        for file_name in content:
            if file_name.endswith(".py"):
                create_file(dir_path, file_name, "w")
            elif file_name.endswith(".md"):
                create_file(dir_path, file_name, "w")
            elif file_name == "requirements.txt":
                create_file(base_dir, file_name, "w")
            elif file_name == "LICENSE":
                create_file(base_dir, file_name, "w")
            elif file_name == "README.md":
                create_file(dir_path, file_name, "w")

# Function to create a basic file
def create_file(directory, file_name, mode):
    file_path = os.path.join(directory, file_name)
    with open(file_path, mode) as f:
        f.write("")  # You can add content to the files here if necessary
    print(f"Created file: {file_path}")

# Create the project structure
project_base_dir = os.getcwd()  # You can change this to any path where you want to create the project
create_project_structure(project_base_dir, project_structure)

print("Project structure created successfully!")
