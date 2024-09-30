
# Milestone 1-6

## `app.py`
A Chainlit app that builds a movies assistant using a custom function calling mechanism. It is generaly broken down into the following functions:

- `on_message()` - Handler for new messages from the user. Calls `generate_response` to generate a response to send to the user.

- `function_calling()` - In a separate conversation specifically designed to only respond in JSON, the main conversation between the user and assistant is sent as context to prompt the special LLM on what function to call next. The response is passed to `process_function_call_response` to execute the function and collect the necessary context to forward back to the main conversation (as a system message).

- `process_function_call_response()` - Processes the function call JSON response to determine the function that needs to be called and to extract the parameters. The specified function is executed and the result is returned as context for the main conversation. The function call JSON is designed to indicate when more information is required to determine the next function to call or the parameter(s) to pass to a function.

# Milestone 7

## `app_using_openai.py`
This replaces the custom function calls code that uses OpenAI's function call feature in its chat completion API, following the guide [here](https://platform.openai.com/docs/guides/function-calling).

# Getting Started

### 1. Create a virtual environment

First, create a virtual environment to isolate the project dependencies:
```bash
python -m venv .venv
```

### 2. Activate the virtual environment:

- On Windows:
  ```bash
  .venv\Scripts\activate
  ```
- On macOS and Linux:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

Install the project dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

- Copy the `.env.sample` file to a new file named `.env`
- Fill in the `.env` file with your API keys

## Running the app

To run the app, use the following command:

```bash
chainlit run app.py -w
``` 

## Updating dependencies

If you need to update the project dependencies, follow these steps:

1. Update the `requirements.in` file with the new package or version.

2. Install `pip-tools` if you haven't already:
   ```bash
   pip install pip-tools
   ```

3. Compile the new `requirements.txt` file:
   ```bash
   pip-compile requirements.in
   ```

4. Install the updated dependencies:
   ```bash
   pip install -r requirements.txt
   ```

This process ensures that all dependencies are properly resolved and pinned to specific versions for reproducibility.
