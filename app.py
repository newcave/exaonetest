import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

# Streamlit app
st.title("EXAONE AI Assistant")
st.write("Interact with the EXAONE model by entering a prompt below:")

# Input field for the prompt
prompt = st.text_input("Enter your prompt", value="Explain who you are")

# Generate response when button is clicked
if st.button("Generate Response"):
    with st.spinner("Generating..."):
        messages = [
            {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Generate output
        output = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128
        )

        # Decode and display the output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("**Response:**")
        st.write(response)
