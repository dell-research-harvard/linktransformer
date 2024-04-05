import streamlit as st
import pandas as pd
import linktransformer as lt

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

st.title('Merge Dataframes using LinkTransformer')
st.write('LinkTransformer supports several AI-powered data wrangling operations - here is an example that allows you to use LLMs to merge data.')

# Function to load DataFrame
def load_dataframe(upload):
    ##if csv is uploaded use read_csv to load the data , otherwise use read_excel
    if upload is not None:
        if upload.name.endswith('csv'):
            return pd.read_csv(upload)
        else:
            return pd.read_excel(upload)
    else:
        return pd.DataFrame()
# Options for DataFrame 1
df1_upload = st.file_uploader("Upload DataFrame 1 (CSV)", type=['csv'], key='df1_upload')

# Options for DataFrame 2
df2_upload = st.file_uploader("Upload DataFrame 2 (CSV)", type=['csv'], key='df2_upload')

# Load and display the DataFrames
df1 = load_dataframe(df1_upload)
df2 = load_dataframe(df2_upload)

if df1 is not None:
    st.write("DataFrame 1 Preview:")
    st.dataframe(df1.head())

if df2 is not None:
    st.write("DataFrame 2 Preview:")
    st.dataframe(df2.head())


# Model selection
model_path = st.text_input("Model path (HuggingFace)", value="all-MiniLM-L6-v2")
st.write("We have trained several record linkage models! Just copy the Hugging Face model path from the [model zoo](https://linktransformer.github.io/).")
##More on model selection available on https://linktransformer.github.io/

if df1_upload is not None and df2_upload is not None:
    # Checkbox for columns to match on
    if not df1.empty and not df2.empty:
        columns_df1 = df1.columns.tolist()
        columns_df2 = df2.columns.tolist()
        selected_columns_df1 = st.multiselect("Select columns from DataFrame 1 to match on:", columns_df1, default=columns_df1[0])
        selected_columns_df2 = st.multiselect("Select columns from DataFrame 2 to match on:", columns_df2, default=columns_df2[0])


         # Perform merge
        if st.button("Merge DataFrames"):
            model=lt.LinkTransformer(model_path)
            df_lm_matched = lt.merge(df1, df2, merge_type='1:m', on=None, model=model, left_on=selected_columns_df1, right_on=selected_columns_df2)
            st.write("Merged DataFrame Preview:")
            st.dataframe(df_lm_matched.head())

            # Download button for merged DataFrame
            csv = convert_df_to_csv(df_lm_matched)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name='merged_dataframe.csv',
                mime='text/csv',
            )
    else:
        st.write("It appears that your dataframes are empty. Please upload valid dataframes.")

       
else:
    st.write("Please upload or enter paths for both DataFrames.")
            
##Add website and citation
st.write("Note that this space only supports CPU usage and is only recommended on small datasets. If you have access to the GPU, check out our python [package](https://github.com/dell-research-harvard/linktransformer/)!")
st.write("For more information and advanced usage, please visit the [LinkTransformer website](https://linktransformer.github.io/).")
st.write("If you use LinkTransformer in your research, please cite the following paper: [LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models](https://arxiv.org/abs/2309.00789)")

