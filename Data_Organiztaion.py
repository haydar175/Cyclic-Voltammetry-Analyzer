import streamlit as st
import pandas as pd
import os
import zipfile
import tempfile
import shutil
from io import BytesIO

def process_and_zip_files(uploaded_zip_file):
    """
    Extracts a zip file to a temporary location, runs the cleaning process,
    and zips the cleaned results for download.
    """
    # Create temporary directories for source and output
    with tempfile.TemporaryDirectory() as temp_source_dir, tempfile.TemporaryDirectory() as temp_output_dir:
        
        # --- 1. Extract the uploaded ZIP to the temporary source directory ---
        with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_source_dir)

        # --- 2. Find all CSV files to process for the progress bar ---
        csv_files_to_process = []
        for root, _, files in os.walk(temp_source_dir):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    csv_files_to_process.append(os.path.join(root, filename))
        
        total_files = len(csv_files_to_process)
        if total_files == 0:
            st.warning("⚠️ No CSV files were found inside the uploaded ZIP file.")
            return None

        # --- 3. Setup Streamlit progress elements ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("Processing Logs", expanded=True)

        # --- 4. Main Processing Loop (using your original logic) ---
        for i, input_path in enumerate(csv_files_to_process):
            filename = os.path.basename(input_path)
            root = os.path.dirname(input_path)
            
            progress_percentage = (i + 1) / total_files
            progress_bar.progress(progress_percentage)
            status_text.info(f"🔄 Processing file {i+1}/{total_files}: {filename}")

            try:
                # Create corresponding output directory
                relative_path = os.path.relpath(root, temp_source_dir)
                output_dir = os.path.join(temp_output_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)

                log_container.write(f"--- Processing: {filename} ---")

                # Manual File Reading and Cleaning
                with open(input_path, "r", encoding="utf-16", errors='ignore') as infile:
                    lines = infile.readlines()
                
                content_lines = lines[6:-2]
                
                processed_data = []
                for line in content_lines:
                    cleaned_line = line.strip().strip('"')
                    if cleaned_line:
                        processed_data.append(cleaned_line.split(","))
                
                if not processed_data:
                    log_container.warning(f"File '{filename}' is empty after cleaning. Skipping.")
                    continue

                # Pandas Processing
                df = pd.DataFrame(processed_data)
                df.dropna(how='any', inplace=True)
                df.dropna(axis=1, how='all', inplace=True)
                
                # Save the final cleaned DataFrame
                df.to_csv(output_path, index=False, header=False, sep=',', encoding="utf-8")
                
                log_container.success(f"✅ Cleaned: {filename}")

            except Exception as e:
                log_container.error(f"❌ Error processing {filename}: {e}")

        status_text.success(f"🎉 All done! Processed {total_files} files.")
        st.balloons()

        # --- 5. Create the output ZIP file from the temporary output directory ---
        output_zip_buffer = BytesIO()
        with zipfile.ZipFile(output_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_output_dir)
                    zipf.write(file_path, arcname)
        
        output_zip_buffer.seek(0)
        return output_zip_buffer


# --- Streamlit App Interface ---

st.set_page_config(page_title="CSV File Processor", layout="wide")

st.title("📂 CSV Folder Processor")
st.markdown("""
This app automates the cleaning of CSV files within a folder structure.
1.  **ZIP your source folder** containing all the raw CSV files and subfolders.
2.  **Upload the ZIP file** below.
3.  The app will process all files, maintaining the original folder structure.
4.  **Download the resulting ZIP file** containing the cleaned data.
""")

# --- User Upload ---
st.header("1. Upload Your Data Folder")
uploaded_zip = st.file_uploader(
    "Choose a ZIP file", 
    type="zip"
)

# --- Processing and Download Logic ---
if uploaded_zip is not None:
    st.header("2. Processing Results")
    with st.spinner('Extracting, cleaning, and re-zipping files... Please wait.'):
        result_zip_buffer = process_and_zip_files(uploaded_zip)

    if result_zip_buffer:
        st.header("3. Download Cleaned Data")
        st.download_button(
            label="🚀 Download Cleaned_Data.zip",
            data=result_zip_buffer,
            file_name="Cleaned_Data.zip",
            mime="application/zip",
            type="primary"
        )
