import streamlit as st
st.set_page_config(page_title="Chat Model")

st.markdown("# Chat Model")
st.sidebar.header("Chat Model")

def get_chat_model(question):
    import torch
    from transformers import pipeline

    generate_text = pipeline(
        model="Ketak-ZoomRx/Drug_Prompt_Ollama_67k",
        torch_dtype="auto",
        trust_remote_code=True,
        use_fast=True,
        device_map={"": "cuda:0"},
    )

    res = generate_text(
    question,
    min_new_tokens=2,
    max_new_tokens=256,
    do_sample=False,
    num_beams=1,
    temperature=float(0.0),
    repetition_penalty=float(1.2),
    renormalize_logits=True)

    return generate_text


def main():
    question =  st.text_input(label='Question', 
                               placeholder='Type your question here',key='question')+'?'
    # Wait until the question is not empty
    if not question:
        st.info("Please enter a question...")

    if question :
        # Run the selected function based on the user's choice
        answer = get_chat_model(question)
        # Once text input is provided, display it
        st.write(answer)

if __name__ == "__main__":
    main()