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
    "functions": ["get_showtimes('The Batman', 'New York')"]
}

If a function needs to be called, the response should only contain JSON data with the function name. For example, if the user asks for now playing movies, the response should be in the following:
{
    "functions": ["get_now_playing_movies()"]
}

If there are multiple functions to call, the response should contain multiple function names in the list:
{
    "functions": ["get_now_playing_movies()", "get_showtimes('The Batman', '95112')"]
}

If multiple functions need to be called, but more information is requiered first, append the callback to request the information. For example, if the user asks for now playing movies and showtimes for any random movie, append the callback function to request the information first:
{
    "functions": ["get_now_playing_movies()", "callback()"]
}

Then, when the required information is provided, the follow-up response should call the next function with the required information. Remember to remove 'get_now_playing_movies()' from the list of functions to call once the information is provided and the next function(s) are being decided:
{
    "functions": ["get_showtimes('The Batman', '95112')"]
}

If there is no appropriate function to call, "functions" should be set to an empty array (i.e., []).
"""

def parse_function_signatures(function_signatures):
    result = []
    for signature in function_signatures:
        # Remove the parentheses and split by the first '('
        func_name, params = signature.split('(', 1)
        # Remove the closing parenthesis and split by ','
        params = params.rstrip(')').split(', ')
        result.append((func_name, params))

    return result

async def process_completion(function_call_history, completion):
    func_json = json.loads(completion.choices[0].message.content)
    function_signatures = func_json['functions']
    functions = parse_function_signatures(function_signatures)
    print("Functions to Call: ", functions)
    if functions.count == 0:
        return None

    context = ""
    for func_name, params in functions:
        if func_name == "get_now_playing_movies":
            context += movie_functions.get_now_playing_movies()
        elif func_name == "get_showtimes":
            title = params[0]
            location = params[1]
            print("Title: ", title)
            print("Location: ", location)
            context += movie_functions.get_showtimes(title, location)
        elif func_name == "get_reviews":
            movie_id = params[0]
            print("Movie ID: ", movie_id)
            context += movie_functions.get_reviews(movie_id)
        elif func_name == "callback":
            function_call_history.append({"role": "system", "content": f"Here's the requested callback with additional information: {context} \n\n Please use this information to decide the next function(s) to call."})
            completion = await client.chat.completions.create(messages=function_call_history, **gen_kwargs)
            context += await process_completion(function_call_history, completion)
    
    return context

async def function_calling(client, message_history):
    # Function calling
    function_call_history = [{"role": "system", "content": FUNCTION_SYSTEM_PROMPT}]
    function_call_history.append({"role": "user", "content": message_history[-1]["content"]})
    completion = await client.chat.completions.create(messages=function_call_history, **gen_kwargs)
    
    try:
        context = await process_completion(function_call_history, completion)

        return context
                                
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
