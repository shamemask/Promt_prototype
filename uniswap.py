from question_model import get_qa

qa = get_qa()
def answer_uniswap_question(question: str):
    if question == "":
        return ""
    response = qa({"question": question})
    print('get qa')
    return response["answers"]


