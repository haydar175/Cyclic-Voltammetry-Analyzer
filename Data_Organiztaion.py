import streamlit as st
import pandas as pd
import os

def process_csv_files(source_folder, output_folder):
    """
    The core processing logic from your original script, adapted to yield
    status updates for the Streamlit interface.
    """
    # --- 1. Find all CSV files first to calculate total for progress bar ---
    csv_files_to_process = []
    for root, _, files in os.walk(source_folder):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_files_to_process.append(os.path.join(root, filename))
    
    total_files = len(csv_files_to_process)
    if total_files == 0:
        st.warning("⚠️ No CSV files found in the specified source folder.")
        return

    # --- 2. Setup Streamlit progress elements ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.expander("Processing Logs", expanded=True)

    # --- 3. Main Processing Loop ---
    for i, input_path in enumerate(csv_files_to_process):
        filename = os.path.basename(input_path)
        root = os.path.dirname(input_path)
        
        # Update progress bar and status text for the current file
        progress_percentage = (i + 1) / total_files
        progress_bar.progress(progress_percentage)
        status_text.info(f"🔄 Processing file {i+1}/{total_files}: {filename}")

        try:
            # --- Create corresponding output directory ---
            relative_path = os.path.relpath(root, source_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)

            log_container.write(f"--- Processing: {filename} ---")

            # --- Stage 1: Manual File Reading and Cleaning ---
            with open(input_path, "r", encoding="utf-16") as infile:
                lines = infile.readlines()
            
            initial_row_count = len(lines)
            content_lines = lines[6:-2]  # Skip header and footer
            
            processed_data = []
            for line in content_lines:
                cleaned_line = line.strip().strip('"')
                if cleaned_line:
                    split_row = cleaned_line.split(",")
                    processed_data.append(split_row)
            
            rows_after_skips = len(processed_data)

            if not processed_data:
                log_container.warning(f"File '{filename}' is empty after skipping header/footer. Skipping.")
                continue

            # --- Stage 2: Pandas Processing ---
            df = pd.DataFrame(processed_data)
            df.dropna(how='any', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            
            final_row_count = len(df)

            # Save the final cleaned DataFrame
            df.to_csv(output_path, index=False, header=False, sep=',', encoding="utf-8")
            
            # Log detailed results to the expander
            log_container.write(f"  - Initial rows: {initial_row_count}")
            log_container.write(f"  - After skipping header/footer: {rows_after_skips}")
            log_container.write(f"  - Final rows after cleanup: {final_row_count}")
            log_container.success(f"✅ Saved to: {output_path}")

        except Exception as e:
            log_container.error(f"❌ Error processing {filename}: {e}")

    status_text.success(f"🎉 All done! Processed {total_files} files.")
    st.balloons()


# --- Streamlit App Interface ---

st.set_page_config(page_title="CSV File Processor", layout="wide")

st.title("📂 CSV File Processor")
st.markdown("""
This app automates the cleaning of CSV files from a specified folder structure.
1.  Enter the full path to the main **source folder** containing your raw CSV files.
2.  Enter the full path to the **output folder** where cleaned files will be saved.
3.  Click the **Start Processing** button.
The app will maintain the original subfolder structure in the output directory.
""")

# --- User Inputs ---
st.header("1. Configure Paths")
source_folder = st.text_input(
    "Enter the Source Folder Path:", 
    "C:/Users/YourUser/Desktop/Raw_Data"
)
output_folder = st.text_input(
    "Enter the Output Folder Path:", 
    "C:/Users/YourUser/Desktop/Cleaned_Data"
)

# --- Action Button and Logic ---
st.header("2. Run the Processor")
if st.button("🚀 Start Processing", type="primary"):
    # Validate source path
    if not os.path.isdir(source_folder):
        st.error("The Source Folder path is not a valid directory. Please check it.")
    elif not source_folder or not output_folder:
        st.warning("Please provide both a source and an output folder path.")
    else:
        try:
            # ✨ NEW: Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            with st.spinner('Processing files...'):
                process_csv_files(source_folder, output_folder)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")