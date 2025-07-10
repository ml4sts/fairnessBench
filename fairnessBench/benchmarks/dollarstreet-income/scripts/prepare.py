import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlcommons/the-dollar-street-dataset")

print("Path to dataset files:", path)
    