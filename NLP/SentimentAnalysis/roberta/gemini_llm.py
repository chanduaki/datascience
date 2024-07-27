import google.generativeai as genai

genai.configure(api_key="GOOGLE_API_KEY")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 5000,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
def get_gemini_gen_ai_summarised_data(input_data):

    prompt_parts =f"""
    ## Task: Summarize input data into a table format for Streamlit display.

    **Args:**
    * input_data: The input data to be summarized.

    **Returns:**
    * The summarized data in table format.

    **Input data:**
    {input_data}

    **Desired output format:**
    | Feedback Category | Number of Feedbacks | Summary of Feedbacks |
    |---|---|---|
    """
    print(prompt_parts)

    response = model.generate_content(prompt_parts)
    output = response.text
    print(output)
    return output

def get_areas_of_improvement(input_data):
    # Define the prompt for summarizing areas of improvement
    prompt_parts = f"""
    ## Task: Identify and summarize areas of improvement based on input feedback data.

    **Args:**
    * input_data: The feedback data to be analyzed.

    **Returns:**
    * A summary of identified areas of improvement in table format.

    **Input data:**
    {input_data}

    **Desired output format:**
    | Area of Improvement | Description |
    |---|---|
    """

    print(prompt_parts)

    # Assuming you have a model that can generate content based on the prompt
    response = model.generate_content(prompt_parts)
    output = response.text
    print(output)
    return output


# Example usage
# input_data_example = "Some feedback data goes here."
# prompt_example = get_gemini_gen_ai_summarised_data(input_data_example)
# print(prompt_example)











