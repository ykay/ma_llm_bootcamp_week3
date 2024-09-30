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

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_now_playing_movies",
            "description": "Get a list of movies that are playing now. Call this whenever you need to know the current movies playing in the theatres, for example when a customer asks 'What movies are currently playing?'",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_showtimes",
            "description": "Get the showtimes for a particular movie playing at a certain location. Call this whenever you need to know the showtimes for a specific movie at a specific location, for example when a customer asks 'What are the showtimes for The Batman in New York?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the movie for which you want to get the showtimes."
                    },
                    "location": {
                        "type": "string",
                        "description": "The zip code for which you want to get the showtimes."
                    }
                },
                "required": ["title", "location"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": "Get the reviews for a particular movie. Call this whenever you need to know the reviews for a specific movie, for example when a customer asks 'What are critics saying about The Batman?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "string",
                        "description": "The TMDB movie ID of the movie for which you want to get the reviews."
                    }
                },
                "required": ["movie_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buy_ticket",
            "description": "Buy a ticket for a particular movie at a specific location and time. Call this whenever you need to buy a ticket for a customer, for example when a customer asks an you buy me a ticket for The Batman at AMC Metreon at 7pm?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "theater": {
                        "type": "string",
                        "description": "The name of the theater where the movie is playing."
                    },
                    "movie": {
                        "type": "string",
                        "description": "The title of the movie for which you want to buy the ticket."
                    },
                    "showtime": {
                        "type": "string",
                        "description": "The time at which the movie is playing."
                    }
                },
                "required": ["theater", "movie", "showtime"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_ticket_purchase",
            "description": "Confirm the purchase of a ticket for a particular movie at a specific location and time. Call this whenever you need to confirm the purchase of a ticket for a customer, for example when a customer asks 'Yes, I want to buy a ticket for The Batman at AMC Metreon at 7pm.'",
            "parameters": {
                "type": "object",
                "properties": {
                    "theater": {
                        "type": "string",
                        "description": "The name of the theater where the movie is playing."
                    },
                    "movie": {
                        "type": "string",
                        "description": "The title of the movie for which you want to buy the ticket."
                    },
                    "showtime": {
                        "type": "string",
                        "description": "The time at which the movie is playing."
                    }
                },
                "required": ["theater", "movie", "showtime"],
                "additionalProperties": False
            }
        }
    },
]

async def process_function_call_response(completion, message_history):
    # This code assumes we have already determined that the model generated a function call.
    tool_call = completion.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print("Function to Call: ", func_name)

    context = ""
    function_call_result_message = None
    if func_name == "get_now_playing_movies":
        print("Calling get_now_playing_movies()")
        context = movie_functions.get_now_playing_movies()
        context_label = "now_playing_movies"
    elif func_name == "get_showtimes":
        title = arguments['title']
        location = arguments['location']
        print("Calling get_showtimes()")
        print("Title: ", title)
        print("Location: ", location)
        context = movie_functions.get_showtimes(title, location)
        context_label = "showtimes"
    elif func_name == "get_reviews":
        movie_id = arguments['movie_id']
        print("Calling get_reviews()")
        print("Movie ID: ", movie_id)
        context = movie_functions.get_reviews(movie_id)
        context_label = "reviews"
    elif func_name == "buy_ticket":
        theater = arguments['theater']
        movie = arguments['movie']
        showtime = arguments['showtime']
        print("Calling buy_ticket(); But confirming first.")
        print("Theater: ", theater)
        print("Movie: ", movie)
        print("Showtime: ", showtime)
        context = f"Ask the user if they really want to buy the ticket for {movie} at {theater} on {showtime}. If they confirm, call confirm_ticket_purchase()."
        context_label = "need_confirmation_to_buy_ticket"
    elif func_name == "confirm_ticket_purchase":
        theater = arguments['theater']
        movie = arguments['movie']
        showtime = arguments['showtime']
        print("Calling confirm_ticket_purchase()")
        print("Theater: ", theater)
        print("Movie: ", movie)
        print("Showtime: ", showtime)
        context = movie_functions.buy_ticket(theater, movie, showtime)
        context = "Just pretend you bought a ticket." # Without this, the assistant responds that it can't buy tickets.
        context_label = "ticket_purchase_confirmation"
    else:
        print("No function matched the function name.")

    if context:
        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({context_label: context}),
            "tool_call_id": completion.choices[0].message.tool_calls[0].id
        }
        message_history.append(function_call_result_message)

async def function_calling(client, message_history):
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        tools=tools,
    )

     # Check if the model has made a tool_call. This is the case either if the "finish_reason" is "tool_calls" or if the "finish_reason" is "stop" and our API request had forced a function call
    if completion.choices[0].finish_reason == "tool_calls":
        print("Model requested a tool call.")
        message_history.append(completion.choices[0].message)

        # Handle tool call
        try:
            await process_function_call_response(completion, message_history)
                                
        except Exception as e:
            print("Unexpected Error: ", e)
            
    elif completion.choices[0].finish_reason == "stop":
        print("Model stopped.")

    else:
        print("Model completed normally. No function call was requested.")

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

    await function_calling(client, message_history)

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
