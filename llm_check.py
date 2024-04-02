import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Print the answer
def intelBert(question,context):

    tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
    model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
    # Tokenize the context and question
    tokens = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)

    # Get the input IDs and attention mask
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Perform question answering
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the start and end positions of the answer
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

    return answer

def askingRoberta(question,context):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    answer = nlp(QA_input)['answer']
    
    return answer


def main():
    st.write(""" # DOCUMENT CONTEXT MODEL """)
    context = st.text_input(label='Context Article', 
                               placeholder='Paste your article here',key='context')
    if not context:
        st.info("Please enter some context...")

    st.write(context)

    question =  st.text_input(label='Question', 
                               placeholder='Type your question here',key='question')+'?'
    # Wait until the question is not empty
    if not question:
        st.info("Please enter a question...")

    option_model = st.selectbox(
    'Choose model',
    ('Intel Dynamic TinyBert','Deepset Roberta'))

    answer = 'None'    
    if question and context and option_model:
        # Run the selected function based on the user's choice
        if option_model == "'Intel Dynamic TinyBert'":
            answer = intelBert(question,context)
        elif option_model == "Deepset Roberta":
            answer = askingRoberta(question,context)
        # Once text input is provided, display it
        st.write(answer)

if __name__ == "__main__":
    main()
