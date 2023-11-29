import os
import autogen


from uniswap import answer_uniswap_question
from utils import parse_token_usage
import time




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
        name="Копирайтер",
        max_consecutive_auto_reply=3, llm_config=llm_copywriter,
        system_message=f"Создаёт рекламу для какого либо продукта",
    )
    marketer = autogen.UserProxyAgent(name="Маркетолог",
                             max_consecutive_auto_reply=3,
                             llm_config=llm_marketer,
                             system_message="Анализирует созданную рекламу Копирайтером и дайт рекомендацию согласно full_documents")
    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        llm_config=llm_copywriter,
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "scratch/coding",
            "use_docker": False
        },
        system_message=f"Копирайтер напиши рекламу про {prompt}",function_map={"answer_uniswap_question": answer_uniswap_question},
    )
    groupchat = autogen.GroupChat(agents=[user_proxy,copywriter,marketer],messages=[],max_round=4)
    manager =autogen.GroupChatManager(groupchat=groupchat,llm_config=llm_copywriter)
    start_time = time.time()
    user_proxy.initiate_chat(
        manager,
        message=prompt,
    )

    messages = user_proxy.chat_messages[manager]
    logged_history = autogen.ChatCompletion.logged_history
    autogen.ChatCompletion.stop_logging()
    response = {
        "messages": messages[1:],
        "usage": parse_token_usage(logged_history),
        "duration": time.time() - start_time,
    }
    return str(response).encode(os.getenv("GRADIO_APP_ENCODING", "utf-8"))
