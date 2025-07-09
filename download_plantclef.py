import os
import requests
import tarfile


def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.raw.read())
        print(f"File downloaded successfully to {destination}")
    else:
        print(f"Failed to download file from {url}")


def extract_tar_gz(file_path, extract_to):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"File extracted successfully to {extract_to}")


def main():
    # URLs for the datasets
    urls = {
        "test": "https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TestPackage/PlantCLEF2015TestDataWithAnnotations.tar.gz",
        "train": "https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TrainingPackage/PlantCLEF2015TrainingData.tar.gz",
    }

    # Ask the user if the dataset exists
    dataset_exists = (
        input("Do you have the dataset already? (yes/no): ").strip().lower()
    )

    if dataset_exists == "yes":
        # Ask for the path to the dataset
        dataset_path = input(
            "Please provide the path to the dataset directory: "
        ).strip()
        if os.path.isdir(dataset_path):
            print(f"Dataset directory found at {dataset_path}")
        else:
            print("The provided path does not exist.")
            return
    else:
        print(
            "Warning: The files are large and may take a significant amount of time to download."
        )
        # Ask for the download directory
        download_dir = input(
            "Please provide the directory where you want to download the datasets: "
        ).strip()
        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        # Download and extract datasets
        for key, url in urls.items():
            file_name = url.split("/")[-1]
            file_path = os.path.join(download_dir, file_name)
            download_file(url, file_path)
            extract_tar_gz(file_path, download_dir)
            # Remove the tar.gz file after extraction
            os.remove(file_path)

        print("Datasets downloaded and extracted successfully.")


if __name__ == "__main__":
    main()
