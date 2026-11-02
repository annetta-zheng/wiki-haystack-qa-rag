#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DOWNLOAD_URL = os.environ.get("DATA_DOWNLOAD_URL", "")
DATA_DIR = "data"
ZIP_FILENAME = "enwiki-1gb.zip"
XML_FILENAME = "enwiki-1gb.xml"

def download_and_extract_data():
    if not os.path.exists("src") or not os.path.exists("data"):
        print("ERROR: Please run this script from the repository root")
        print("Expected structure: src/ and data/ should exist")
        sys.exit(1)
    
    data_path = Path(DATA_DIR)
    zip_path = data_path / ZIP_FILENAME
    xml_path = data_path / XML_FILENAME
    
    if xml_path.exists():
        print(f"✓ {XML_FILENAME} already exists in {DATA_DIR}/")
        return
    
    if not DATA_DOWNLOAD_URL:
        print("ERROR: set DATA_DOWNLOAD_URL to a URL of enwiki-1gb.zip,")
        print(f"or place {XML_FILENAME} in {DATA_DIR}/enwiki/ yourself.")
        sys.exit(1)

    print("Downloading Wikipedia data file...")
    print("This may take a few minutes as the file is large...")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rDownloading: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="", flush=True)
        
        urllib.request.urlretrieve(DATA_DOWNLOAD_URL, zip_path, reporthook=show_progress)
        print("\n✓ Download completed!")
        
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        print(f"✓ Extracted to {DATA_DIR}/")
        
        zip_path.unlink()
        print("✓ Cleaned up ZIP file")

    except Exception as e:
        print(f"\nERROR downloading or extracting data: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

if __name__ == "__main__":
    print("=== Wikipedia Data Setup ===")
    download_and_extract_data()
    print("\n🎉 Setup complete! You can now run your solution.")
    print("\nNext steps:")
    print("Run the examples with: cd src && python qa_agent.py")