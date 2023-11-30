from Promt_prototype.question_model import get_qa

# Модуль для предварительной обработки pdf, при запуске программы, для использовании ассистентами
qa = get_qa()
def answer_uniswap_question(question: str):
    if question == "":
        return ""
    response = qa({"question": question})
    print('get qa')
    return response["answers"]


