import os
import autogen

import time

from Promt_prototype.uniswap import answer_uniswap_question


def run_flow(prompt: str) -> str:
    from main import OPENAI_API_KEY
    config_list = [{
        'model': 'gpt-4',
        'api_key': OPENAI_API_KEY,
    }]
    llm_copywriter ={
        "config_list": config_list

    }
    llm_marketer = {
        "seed": 42,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,
        "function": [{
            "name":"answer_uniswap_question",
            "description": "Ответ на любой Uniswape вопрос",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Вопрос, который следует задать согласно uniswap",
                    },
                },
                "required": ["question"],
            },
        }], # temperature for sampling
        "use_cache": True,  # whether to use cache
    }
    copywriter = autogen.AssistantAgent(
        name="Copywriter",
        max_consecutive_auto_reply=3, llm_config=llm_copywriter,
        system_message=f"Copywriter. Ты создаёшь рекламу для какого либо продукта",
    )
    marketer = autogen.AssistantAgent(name="Marketer",
                                      max_consecutive_auto_reply=3,
                                      llm_config=llm_marketer,
                                      system_message="Marketer. Ты анализируешь созданную рекламу Копирайтером и даёшь рекомендацию согласно full_documents")
    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        llm_config=llm_copywriter,
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": ".",
        },
        system_message=f"Admin. Просто следит за общением ассистентов.",
        function_map={"answer_uniswap_question": answer_uniswap_question},
    )
    groupchat = autogen.GroupChat(agents=[user_proxy, copywriter, marketer], messages=[], max_round=3)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_copywriter)
    start_time = time.time()
    user_proxy.initiate_chat(
        manager,
        message=f"""
            1. Copywriter напиши рекламу про {prompt}. 
            2. Marketer проанализируй рекламу Копирайтера и дай рекомендации копирайтеру согласно full_documents
            3. Copywriter измени свой рекламный текст учитывая рекомендации маркетолога.
            4. Marketer снова проанализируй результат Копирайтера и дай рекомендации копирайтеру согласно full_documents.
            5. Copywriter снова измени свой рекламный текст учитывая рекомендации маркетолога.
            6. Marketer снова проанализируй результат Копирайтера и дай рекомендации копирайтеру согласно full_documents.
            7. Copywriter снова измени свой рекламный текст учитывая рекомендации маркетолога.
            8. Marketer снова проанализируй результат Копирайтера и дай рекомендации копирайтеру согласно full_documents.
            """,
    )

    messages = user_proxy.chat_messages[manager]
    messages = '\n'.join([ message["role"]+':\n'+message["content"] for message in messages[1:]])
    return messages
