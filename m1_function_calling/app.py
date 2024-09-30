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

function_signatures = {
    "get_now_playing_movies()",
    "get_showtimes(title, location)",
    "get_reviews(movie_id)",
    "buy_ticket(theater, movie, showtime)",
    "confirm_ticket_purchase(theater, movie, showtime)",
    "callback()"
}

SYSTEM_PROMPT = """
You are a movie guru. You don't provide awkward qualifiers like, "According to TMDB API..." because no one talks like that and you should speak as if you already know what you know.
"""

FUNCTION_SYSTEM_PROMPT = f"""\
You provide information about movies. When you don't have the required information, you need to provide the appropriate function to call based on the user’s needs. When a function needs to be called, respond with one of the following function names:

{function_signatures}

Make sure to replace parameters with the appropriate values. For example, if the user asks for showtimes for the movie "The Batman" in "New York", the response should be in the following:

{{
    "functions": ["get_showtimes('The Batman', 'New York')"]
}}

If a function needs to be called, the response should only contain JSON data with the function name. For example, if the user asks for now playing movies, the response should be in the following:
{{
    "functions": ["get_now_playing_movies()"]
}}

If there are multiple functions to call, the response should contain multiple function names in the list:
{{
    "functions": ["get_now_playing_movies()", "get_showtimes('The Batman', '95112')"]
}}

If multiple functions need to be called, but more information is required first, append the callback to request the information. For example, if the user asks for now playing movies and showtimes for any random movie, append the callback function to request the information first:
{{
    "functions": ["get_now_playing_movies()", "callback()"]
}}

You can also request a specific piece of information. For example, if the user asks for showtimes for a movie, but the location is missing, you can request the location first:
{{
    "functions": ["get_showtimes('The Batman', 'callback()')"]
}}

Then, when the required information is provided, the follow-up response should call the next function with the required information. Remember to remove 'get_now_playing_movies()' from the list of functions to call once the information is provided and the next function(s) are being decided:
{{
    "functions": ["get_showtimes('The Batman', '95112')"]
}}

If there is no appropriate function to call, "functions" should be set to an empty array (i.e., []).
"""

function_call_history = [{"role": "system", "content": FUNCTION_SYSTEM_PROMPT}]

def parse_missing_info(functions):
    context = ""
    for func_name, params in functions:
        for param in params:
            if "callback()" in param:
                print(f"Callback detected in parameters for {func_name}; Requesting more information.")
                # Replace callback() with [Missing Info]
                params[params.index(param)] = "[Missing Info]"
                matching_signature = next(filter(lambda s: func_name in s, function_signatures), None)
                context += f"Following information needed from the user for '{matching_signature}'; {params}\n"
    
    return context

def parse_function_signatures(function_signatures):
    result = []
    for signature in function_signatures:
        # Remove the parentheses and split by the first '('
        func_name, params = signature.split('(', 1)
        # Remove the closing parenthesis and split by ','
        params = params.rstrip(')').split(', ')
        result.append((func_name, params))

    return result

async def process_function_call_response(completion):
    function_call_history.append({"role": "assistant", "content": completion.choices[0].message.content})
    func_json = json.loads(completion.choices[0].message.content)
    function_signatures = func_json['functions']
    functions_to_call = parse_function_signatures(function_signatures)
    print("Functions to Call: ", functions_to_call)
    if functions_to_call.count == 0:
        return None

    context = ""

    # If any of the parameters are missing, mark the missing information and return immediately to request the missing information from the user
    context += parse_missing_info(functions_to_call)
    if context:
        return context

    for func_name, params in functions_to_call:
        if func_name == "get_now_playing_movies":
            print("Calling get_now_playing_movies()")
            context += movie_functions.get_now_playing_movies()
        elif func_name == "get_showtimes":
            title = params[0]
            location = params[1]
            print("Calling get_showtimes()")
            print("Title: ", title)
            print("Location: ", location)
            context += movie_functions.get_showtimes(title, location)
        elif func_name == "get_reviews":
            movie_id = params[0]
            print("Calling get_reviews()")
            print("Movie ID: ", movie_id)
            context += movie_functions.get_reviews(movie_id)
        elif func_name == "buy_ticket":
            theater = params[0]
            movie = params[1]
            showtime = params[2]
            print("Calling buy_ticket(); But confirming first.")
            print("Theater: ", theater)
            print("Movie: ", movie)
            print("Showtime: ", showtime)
            context += f"Ask the user if they really want to buy the ticket for {movie} at {theater} on {showtime}. If they confirm, call confirm_ticket_purchase()."
        elif func_name == "confirm_ticket_purchase":
            theater = params[0]
            movie = params[1]
            showtime = params[2]
            print("Calling confirm_ticket_purchase()")
            print("Theater: ", theater)
            print("Movie: ", movie)
            print("Showtime: ", showtime)
            context += movie_functions.buy_ticket(theater, movie, showtime)
            context += "Just pretend you bought a ticket." # Without this, the assistant responds that it can't buy tickets.
        elif func_name == "callback":
            if context:
                print("Invoking callback with additional context.")
                function_call_history.append({"role": "system", "content": f"Here's the requested callback with additional information: {context} \n\n Please use this information to decide the next function(s) to call."})
                completion = await client.chat.completions.create(messages=function_call_history, **gen_kwargs)
                context += await process_function_call_response(completion)
            else:
                print("No context to provide callback; Ignoring callback request.")
    
    return context

async def function_calling(client, message_history):
    # Append the last message from the user
    function_call_history.append({"role": "system", "content": f"Conversation Between User and Assistant: {message_history}"})
    completion = await client.chat.completions.create(messages=function_call_history, **gen_kwargs)
    
    try:
        context = await process_function_call_response(completion)

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
