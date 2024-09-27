from dotenv import load_dotenv
import chainlit as cl
import json
import movie_functions

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """
You are a movie guru. You don't provide awkward qualifiers like, "According to TMDB API..." because no one talks like that and you should speak as if you already know what you know.
"""

FUNCTION_SYSTEM_PROMPT = """\
You provide information about movies. When you don't have the required information, you need to provide the appropriate function to call based on the user’s needs. When a function needs to be called, respond with one of the following function names:

get_now_playing_movies()
get_showtimes(title, location)
get_reviews(movie_id)
NONE

Make sure to replace parameters with the appropriate values. For example, if the user asks for showtimes for the movie "The Batman" in "New York", the response should be in the following:

{
    "function": "get_showtimes('The Batman', 'New York')"
}

If a function needs to be called, the response should only contain JSON data with the function name. For example, if the user asks for now playing movies, the response should be in the following:
{
    "function": "get_now_playing_movies()"
}

If there is no appropriate function to call, "function" should be set to "NONE".
"""

def parse_function_signature(signature):
    # Remove the parentheses and split by the first '('
    func_name, params = signature.split('(', 1)
    # Remove the closing parenthesis and split by ','
    params = params.rstrip(')').split(', ')

    return func_name, params

async def function_calling(client, message_history):
    # Function calling
    function_call_history = [{"role": "system", "content": FUNCTION_SYSTEM_PROMPT}]
    function_call_history.append({"role": "user", "content": message_history[-1]["content"]})
    completion = await client.chat.completions.create(messages=function_call_history, **gen_kwargs)
    
    try:
        func_json = json.loads(completion.choices[0].message.content)
        function_call = func_json['function']
        print("Function Call: ", function_call)
        if function_call == "NONE":
            return None
        
        func_name, params = parse_function_signature(function_call)

        print("Function: ", func_name)
        print("Parameters: ", params)

        if func_name == "get_now_playing_movies":
            return movie_functions.get_now_playing_movies()
        elif func_name == "get_showtimes":
            title = params[0]
            location = params[1]
            print("Title: ", title)
            print("Location: ", location)
            return movie_functions.get_showtimes(title, location)
        elif func_name == "get_reviews":
            movie_id = params[0]
            print("Movie ID: ", movie_id)
            return movie_functions.get_reviews(movie_id)
                                
    except Exception as e:
        print("Unexpected Error: ", e)
    
    return None

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    # Start the indicator that the assistant is typing
    response_message = cl.Message(content="")
    await response_message.send()

    if context := await function_calling(client, message_history):
        message_history.append({"role": "system", "content": context})
    else:
        print("No function call")

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
