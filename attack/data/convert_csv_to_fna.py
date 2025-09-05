# convert the csv file with assession id into fna file


import os
import pandas as pd
import subprocess
import time
import argparse

# Function to download sequences in batches
def download_sequences_batch(accession_list, batch_size=500, output_dir="ncbi_downloads"):
    """
    Download sequences from NCBI using the datasets command in batches.
    
    Parameters:
    - accession_list: List of accession IDs
    - batch_size: Number of accessions to download per batch
    - output_dir: Directory to store the downloaded data
    """
    
    # Convert to absolute path and create output directory if it doesn't exist
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches to avoid command line length limits
    total_batches = (len(accession_list) + batch_size - 1) // batch_size
    
    print(f"\nProcessing {len(accession_list)} accessions in {total_batches} batches...")
    
    for i in range(0, len(accession_list), batch_size):
        batch_num = i // batch_size + 1
        batch = accession_list[i:i + batch_size]
        
        # Join accession IDs with spaces for the command
        accession_string = " ".join(batch)
        
        # Construct the datasets command
        cmd = f"/workspaces/BioRiskEval/attack/data/datasets download virus genome accession {accession_string}"
        
        print(f"\nBatch {batch_num}/{total_batches}: Downloading {len(batch)} sequences...")
        
        try:
            # Execute the command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode == 0:
                print(f"✓ Batch {batch_num} downloaded successfully")
                
                # rename the zip file from ncbi_dataset.zip to batch_{batch_num}.zip
                source_zip = os.path.abspath(os.path.join(output_dir, "ncbi_dataset.zip"))
                dest_zip = os.path.abspath(os.path.join(output_dir, f"batch_{batch_num}.zip"))
                os.rename(source_zip, dest_zip)
                
                # Unzip the downloaded file with overwrite flag to avoid prompts
                zip_path = dest_zip
                if os.path.exists(zip_path):
                    # Use -o flag to overwrite without prompting, -q for quiet mode
                    unzip_cmd = f"unzip -o -q {zip_path}"
                    subprocess.run(unzip_cmd, shell=True, cwd=output_dir)
                    print(f"✓ Batch {batch_num} extracted")
                    
                    # Remove the zip file to save space
                    os.remove(zip_path)
                    source_folder = os.path.abspath(os.path.join(output_dir, f"ncbi_dataset"))
                    dest_folder = os.path.abspath(os.path.join(output_dir, f"batch_{batch_num}"))
                    os.rename(source_folder, dest_folder)
                else:
                    print(f"⚠️ Warning: Expected zip file not found for batch {batch_num}")
                
            else:
                print(f"✗ Error downloading batch {batch_num}: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Exception occurred for batch {batch_num}: {str(e)}")
        
        # Small delay between batches to be respectful to the server
        if batch_num < total_batches:
            time.sleep(2)
    
    print(f"\n✓ Download process completed. Check the '{output_dir}' directory for results.")
    
    # List the contents of the output directory
    print(f"\nContents of {output_dir}:")
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")

def merge_all_batches(base_dir="/workspaces/BioRiskEval/attack/ncbi_downloads_all"):
    # find all the batch folders in the base_dir
    batch_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("batch_")]
    batch_folders.sort()  # Sort to ensure consistent order
    print(f"Found {len(batch_folders)} batch folders")
    print(f"Batch folders: {batch_folders}")
    # merge all the .fna files into one fna file
    for batch_folder in batch_folders:
        batch_dir = os.path.join(base_dir, batch_folder, "data")
        fna_files = [f for f in os.listdir(batch_dir) if f.endswith(".fna")]
        print(f"Found {len(fna_files)} .fna files in {batch_folder}")
        # merge all the .fna files into one fna file
        with open(os.path.join(base_dir, "merged.fna"), "a") as outfile:
            for fna_file in fna_files:
                with open(os.path.join(batch_dir, fna_file), "r") as infile:
                    outfile.write(infile.read())
        print(f"Merged {len(fna_files)} .fna files into {base_dir}/merged.fna")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to FNA")
    parser.add_argument("--file_name", type=str, required=True, help="The name of the CSV file to convert")
    args = parser.parse_args()
    file_name = args.file_name

    df_test = pd.read_csv(file_name)

    # Extract accession IDs
    accession_ids = df_test['#Accession'].tolist()
    print(f"Total accession IDs: {len(accession_ids)}")

    # Show first 10 accession IDs as example
    print(f"First 10 accession IDs: {accession_ids[:10]}")

    print("\n" + "="*50)
    print("Downloading sequences")
    print("="*50)
    if "ft_dataset" in file_name:
        output_dir = f"ft_dataset/ncbi_downloads_{file_name.split('/')[-1].replace('.csv', '').replace('accession_id_', '')}"
    elif "eval_dataset" in file_name:
        output_dir = f"eval_dataset/ncbi_downloads_{file_name.split('/')[-1].replace('.csv', '').replace('accession_id_', '')}"
    else:
        raise ValueError(f"Invalid file name: {file_name}, must specify ft_dataset or eval_dataset")
    output_dir = os.path.abspath(output_dir)

    download_sequences_batch(accession_ids, batch_size=500, output_dir=output_dir)

    merge_all_batches(output_dir)

