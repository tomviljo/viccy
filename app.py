# Standard library
import html
import json
import os
import random
import re

# Third-party libraries
from dotenv import load_dotenv
import openai
from slack_bolt import App
import slack_sdk
import tiktoken

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

gpt_model = os.environ.get('GPT_MODEL', 'gpt-3.5-turbo-0301')
gpt_max_tokens = 4096  # TODO: GPT-4 support
gpt_overhead_tokens = 3 # Every reply is primed with <|start|>assistant<|message|>, which is 3 tokens
gpt_reply_tokens = int(os.environ.get('GPT_REPLY_TOKENS', '500'))
gpt_temperature = float(os.environ.get('GPT_TEMPERATURE', '0.4'))
gpt_presence_penalty = float(os.environ.get('GPT_PRESENCE_PENALTY', '0.2'))
gpt_frequency_penalty = float(os.environ.get('GPT_FREQUENCY_PENALTY', '0.2'))
gpt_system_message = 'You are Viccy, a helpful assistant based on ChatGPT. Answer as concisely as possible.'

app = App(
    token = os.environ['SLACK_BOT_TOKEN'],
    signing_secret = os.environ['SLACK_SIGNING_SECRET']
)

def sanitize(text):
    # Remove user and channel tags
    text = re.sub(r'<.*?>', '', text)
    # Unescape &lt; and &gt; back to < and >
    text = html.unescape(text)
    # Strip leading and trailing whitespace
    return text.strip()

def escape(text):
    text = re.sub('&', '&amp;', text)
    text = re.sub('<', '&lt;', text)
    return re.sub('>', '&gt;', text)

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text)

def truncate(text, length=50):
    if len(text) > length:
        return text[:length] + '...'
    return text

def snippet(text):
    return truncate(normalize_whitespace(escape(sanitize(text))))

def get_history(channel):
    history = app.client.conversations_history(
        channel = channel,
        include_all_metadata = True,
        limit = 200
    )
    messages = list(reversed(history.get('messages', [])))
    messages_by_ts = {m['ts']: m for m in messages}
    bot_messages = [m for m in messages if m.get('metadata', {}).get('event_type', '').startswith('viccy_')]  # TODO: Check bot_id for authenticity
    response_messages = [m for m in messages if m.get('metadata', {}).get('event_type') == 'viccy_response']  # TODO: Check bot_id for authenticity
    # TODO: Optimize
    recording = True
    pairs = []
    for bot_message in bot_messages:
        event_type = bot_message['metadata']['event_type']
        if event_type == 'viccy_response':
            if not recording:
                continue
            request_ts = bot_message.get('metadata', {}).get('event_payload', {}).get('request_ts')
            if not request_ts:
                continue
            request_message = messages_by_ts.get(request_ts)
            if not request_message:
                continue
            pairs.append((request_message, bot_message))
        elif event_type == 'viccy_start':
            recording = True
        elif event_type == 'viccy_stop':
            recording = False
        elif event_type == 'viccy_reset':
            pairs = []
    return pairs, recording

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def count_message_tokens(message, model):
    """Returns the number of tokens used by a message."""
    encoding = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"num_tokens_from_message() is not implemented for model {model}.")
    num_tokens = tokens_per_message
    for key, value in message.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += tokens_per_name
    return num_tokens

def completion_cost(result, model):
    return result.usage.total_tokens * 0.002 / 1000.0

def handle_request(event, say):
    is_im = event.get('channel_type') == 'im'
    if 'thread_ts' in event:
        sorry = 'Sorry' if is_im else f"Sorry <@{event['user']}>"
        say(
            text = f"{sorry}, I don't reply in threads because I need our chat history to be linear.",
            thread_ts = event['thread_ts']
        )
        return

    system_message = {'role': 'system', 'content': gpt_system_message}
    prompt = sanitize(event['text'])
    prompt_message = {'role': 'user', 'content': prompt}
    prompt_tokens = (
        gpt_overhead_tokens + 
        count_message_tokens(system_message, gpt_model) + 
        count_message_tokens(prompt_message, gpt_model)
    )
    max_prompt_tokens = gpt_max_tokens - gpt_reply_tokens

    pairs, recording = get_history(event['channel'])
    history_messages = []
    for pair in reversed(pairs):  # Start from the latest and work backward
        user_message = {'role': 'user', 'content': sanitize(pair[0]['text'])}
        assistant_message = {'role': 'assistant', 'content': sanitize(pair[1]['text'])}
        pair_tokens = count_message_tokens(user_message, gpt_model) + count_message_tokens(assistant_message, gpt_model)
        if prompt_tokens + pair_tokens > max_prompt_tokens:
            break
        prompt_tokens += pair_tokens
        history_messages.append(assistant_message)  # Put the reply first because we will reverse the list
        history_messages.append(user_message)
    messages = [system_message] + list(reversed(history_messages)) + [prompt_message]
    # print(messages)

    result = openai.ChatCompletion.create(
        model = gpt_model,
        max_tokens = gpt_max_tokens - prompt_tokens,  # Use all leftover space
        temperature = gpt_temperature,
        presence_penalty = gpt_presence_penalty,
        frequency_penalty = gpt_frequency_penalty,
        messages = messages
    )
    response = result.choices[0].message.content
    actual_prompt_tokens = result.usage.prompt_tokens
    completion_tokens = result.usage.completion_tokens if 'completion_tokens' in result.usage else 0
    total_tokens = result.usage.total_tokens
    cost = completion_cost(result, gpt_model)
    print(f'Completion finished: {actual_prompt_tokens}/{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {total_tokens} total tokens, {cost} dollars')

    prefix = '' if is_im else f"<@{event['user']}> "
    say(
        text = prefix + response,
        metadata = {
            'event_type': 'viccy_response',
            'event_payload': {
                'request_ts': event['ts']
            }
        }
    )

@app.event('app_mention')
def handle_mention(event, say):
    print('app_mention:', event)
    handle_request(event, say)

@app.message()
def handle_message(event, say):
    print('message:', event)
    # print('message:', event['text'])
    if event.get('channel_type') == 'im':
        handle_request(event, say)

@app.command('/viccy')
def handle_command(ack, say, command):
    print('command:', command)
    # ack(response_type = 'in_channel')
    ack()

    def ephemeral(text):
        app.client.chat_postEphemeral(
            channel = command['channel_id'],
            user = command['user_id'],
            text = text
        )
    usage = \
        'I understand these commands:\n' \
        '```\n' \
        '/viccy start   Start recording chat history\n' \
        '/viccy stop    Stop recording chat history\n' \
        '/viccy status  Check whether I am recording chat history\n' \
        '/viccy reset   Clear all chat history\n' \
        '/viccy list    Print the chat history\n' \
        '```'
    by_request = '' if command['channel_id'].startswith('D') else f", by request from <@{command['user_id']}>"

    subcommand = command['text']
    if subcommand == 'start':
        say(
            text = f'Starting to record chat history{by_request}.',
            metadata = {
                'event_type': 'viccy_start',
                'event_payload': {}
            }
        )
    elif subcommand == 'stop':
        say(
            text = f'I am no longer recording chat history{by_request}.',
            metadata = {
                'event_type': 'viccy_stop',
                'event_payload': {}
            }
        )
    elif subcommand == 'status':
        pairs, recording = get_history(command['channel_id'])
        if recording:
            ephemeral('Yes, I am recording chat history.')
        else:
            ephemeral('No, I am not recording chat history.')
    elif subcommand == 'reset':
        say(
            text = f'Cleared all chat history{by_request}.',
            metadata = {
                'event_type': 'viccy_reset',
                'event_payload': {}
            }
        )
    elif subcommand == 'list':
        pairs, recording = get_history(command['channel_id'])
        pairs_text = '\n'.join([f"> {snippet(p[0]['text'])}\n{snippet(p[1]['text'])}" for p in pairs])
        if pairs_text:
            ephemeral('This is the chat history:\n' + pairs_text)
        else:
            ephemeral('The chat history is empty.')
    elif subcommand in ['', 'help']:
        ephemeral(usage)
    else:
        ephemeral(f"Sorry, I don't recognize the command `{subcommand}`.\n{usage}")

@app.event('app_home_opened')
def handle_home_opened(client, event, logger):
    print('app_home_opened:', event)
    try:
        client.views_publish(
            # Use the user ID associated with the event
            user_id=event["user"],
            # Home tabs must be enabled in your app configuration
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Hello <@" + event["user"] + ">!*"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                          "type": "mrkdwn",
                          "text": "There is not much to see here yet."
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")

if __name__ == '__main__':
    app.start(4000)
