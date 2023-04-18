# Standard library
import html
import json
import os
import random
import re
import time

# Third-party libraries
from dotenv import load_dotenv
import openai
from slack_bolt import App
import slack_sdk
import tiktoken

load_dotenv()

bot_name = os.environ.get('BOT_NAME', 'viccy')
port = int(os.environ.get('PORT', '8000'))

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

def is_bot_message(message):
    # TODO: Check bot_id for authenticity
    if 'metadata' not in message:
        return False
    if 'event_type' not in message['metadata']:
        return False
    return message['metadata']['event_type'].startswith('viccy_')

def get_history(channel, thread_ts):
    if thread_ts:
        # Get unthreaded messages before thread parent, newest to oldest
        history = app.client.conversations_history(
            channel = channel,
            latest = thread_ts,
            include_all_metadata = True,
            limit = 200
        )
        # Get threaded messages including the thread parent, oldest to newest
        replies = app.client.conversations_replies(
            channel = channel,
            ts = thread_ts,
            include_all_metadata = True,
            limit = 200
        )
        messages = list(reversed(history['messages'])) + replies['messages']
    else:
        # Get unthreaded messages, newest to oldest
        history = app.client.conversations_history(
            channel = channel,
            include_all_metadata = True,
            limit = 200
        )
        messages = list(reversed(history['messages']))
    # for m in messages:
    #     print(f'--- history ts {m["ts"]} thread_ts {m.get("thread_ts")} text {snippet(m["text"])}')
    messages_by_ts = {m['ts']: m for m in messages}
    bot_messages = [m for m in messages if is_bot_message(m)]
    # TODO: Optimize
    recording = True
    system_message = None
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
            system_message = None
            pairs = []
        elif event_type == 'viccy_system':
            content = bot_message.get('metadata', {}).get('event_payload', {}).get('content')
            if not content:
                continue
            system_message = content
    return {
        'pairs': pairs, 
        'bot_messages': bot_messages,
        'system_message': system_message or gpt_system_message,
        'recording': recording
    }

def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

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

def completion_cost(prompt_tokens, response_tokens, model):
    # TODO: GPT-4
    return (prompt_tokens + response_tokens) * 0.002 / 1000.0

def handle_request(event, say):
    history = get_history(event['channel'], event.get('thread_ts'))

    system_message = {'role': 'system', 'content': history['system_message']}
    question = sanitize(event['text'])
    question_message = {'role': 'user', 'content': question}
    prompt_tokens = (
        gpt_overhead_tokens + 
        count_message_tokens(system_message, gpt_model) + 
        count_message_tokens(question_message, gpt_model)
    )
    max_prompt_tokens = gpt_max_tokens - gpt_reply_tokens

    history_messages = []
    for pair in reversed(history['pairs']):  # Start from the latest and work backward
        user_message = {'role': 'user', 'content': sanitize(pair[0]['text'])}
        assistant_message = {'role': 'assistant', 'content': sanitize(pair[1]['text'])}
        pair_tokens = count_message_tokens(user_message, gpt_model) + count_message_tokens(assistant_message, gpt_model)
        if prompt_tokens + pair_tokens > max_prompt_tokens:
            break
        prompt_tokens += pair_tokens
        history_messages.append(assistant_message)  # Put the reply first because we will reverse the list
        history_messages.append(user_message)
    messages = [system_message] + list(reversed(history_messages)) + [question_message]
    # print(messages)

    msg = app.client.chat_postMessage(
        channel = event['channel'],
        thread_ts = event.get('thread_ts'),
        text = '...'
    )
    msg_time = time.time()
    response = openai.ChatCompletion.create(
        model = gpt_model,
        max_tokens = gpt_max_tokens - prompt_tokens,  # Use all leftover space
        temperature = gpt_temperature,
        presence_penalty = gpt_presence_penalty,
        frequency_penalty = gpt_frequency_penalty,
        messages = messages,
        stream = True
    )
    response_text = ''
    for chunk in response:
        chunk_text = chunk['choices'][0]['delta'].get('content', '')
        now = time.time()
        if now > msg_time + 1.0:
            app.client.chat_update(
                channel = event['channel'],
                ts = msg['ts'],
                text = response_text + ' ...'
            )
            msg_time = now
        response_text += chunk_text
    response_tokens = count_tokens(response_text, gpt_model)  # FIXME: Not entirely accurate because we don't count the role header?
    cost = completion_cost(prompt_tokens, response_tokens, gpt_model)
    print(f'Completion finished: {prompt_tokens} prompt tokens, {response_tokens} response tokens, {prompt_tokens + response_tokens} total tokens, {cost} dollars')

    is_im = event.get('channel_type') == 'im'
    suffix = '' if is_im else f" <@{event['user']}>"
    app.client.chat_update(
        channel = event['channel'],
        ts = msg['ts'],
        text = response_text + suffix,
        metadata = {
            'event_type': 'viccy_response',
            'event_payload': {
                'request_ts': event['ts']
            }
        }
    )

@app.event('app_mention')
def handle_mention(event, say):
    print('app_mention event:', event)
    handle_request(event, say)

@app.event('message')
def handle_message(event, say):
    print('message event:', event)
    if event.get('channel_type') == 'im':
        handle_request(event, say)

@app.command(f'/{bot_name}')
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
        f'/{bot_name} start             Start recording chat history\n' \
        f'/{bot_name} stop              Stop recording chat history\n' \
        f'/{bot_name} status            Check whether I am recording chat history\n' \
        f'/{bot_name} system            Print my current system message\n' \
        f'/{bot_name} system [message]  Set my system message\n' \
        f'/{bot_name} undo              Undo the last command or response\n' \
        f'/{bot_name} reset             Clear the chat history and system message\n' \
        f'/{bot_name} list              Print the chat history\n' \
        '```'
    by_request = '' if command['channel_id'].startswith('D') else f", by request from <@{command['user_id']}>"

    cmd_and_arg = command['text'].split(maxsplit = 1)
    cmd = cmd_and_arg[0].lower() if len(cmd_and_arg) else ''
    arg = cmd_and_arg[1] if len(cmd_and_arg) == 2 else ''
    if cmd == 'start' and not arg:
        say(
            text = f'Starting to record chat history{by_request}.',
            metadata = {
                'event_type': 'viccy_start',
                'event_payload': {}
            }
        )
    elif cmd == 'stop' and not arg:
        say(
            text = f'I am no longer recording chat history{by_request}.',
            metadata = {
                'event_type': 'viccy_stop',
                'event_payload': {}
            }
        )
    elif cmd == 'status' and not arg:
        history = get_history(command['channel_id'])
        if history['recording']:
            ephemeral('Yes, I am recording chat history.')
        else:
            ephemeral('No, I am not recording chat history.')
    elif cmd == 'system' and not arg:
        history = get_history(command['channel_id'], command.get('thread_ts'))
        ephemeral(f"This is my current system message:\n>{history['system_message']}")
    elif cmd == 'system' and arg:
        say(
            text = f'Updated my system message{by_request}:\n>{arg}',
            metadata = {
                'event_type': 'viccy_system',
                'event_payload': {
                    'content': arg
                }
            }
        )
    elif cmd == 'undo' and not arg:
        history = get_history(command['channel_id'], command.get('thread_ts'))
        messages = history['bot_messages']
        if len(messages):
            message = messages[len(messages) - 1]
            app.client.chat_delete(
                channel = command['channel_id'],
                ts = message['ts']
            )
            say(
                text = f"Deleted the last command{by_request}:\n>{snippet(message['text'])}"
            )
        else:
            ephemeral('There is nothing to undo.')
    elif cmd == 'reset' and not arg:
        say(
            text = f'Cleared all chat history{by_request}.',
            metadata = {
                'event_type': 'viccy_reset',
                'event_payload': {}
            }
        )
    elif cmd == 'list' and not arg:
        history = get_history(command['channel_id'], command.get('thread_ts'))
        pairs_text = '\n'.join([f"> {snippet(p[0]['text'])}\n{snippet(p[1]['text'])}" for p in history['pairs']])
        if pairs_text:
            ephemeral('This is the chat history:\n' + pairs_text)
        else:
            ephemeral('The chat history is empty.')
    elif cmd == 'gpt' and not arg:
        ephemeral(
            f'I am based on the {gpt_model} model with temperature {gpt_temperature}, presence penalty {gpt_presence_penalty} and frequency penalty {gpt_frequency_penalty}.'
        )
    elif cmd in ['', 'help']:
        ephemeral(usage)
    else:
        ephemeral(f"Sorry, I don't recognize the command `{command['text']}`.\n{usage}")

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
                            "text": (
                                "I am an assistant based on ChatGPT, with the following added features:\n"
                                "\n"
                                "*Group conversations*. Multiple users can participate in chatting with me, just ping me in a channel or group conversation.\n"
                                "\n"
                                "*Threaded conversations*. Reply to me in a thread to go off on a tangent.\n"
                                "\n"
                                f"*Custom roles*. Use `/{bot_name} system` to view or change my underlying instructions in the context of a conversation.\n"
                                "\n"
                                f"*Freeze and unfreeze history*. Establish a context through dialogue and turn it into a template for answering questions. Type `/{bot_name}` to learn more.\n"
                                "\n"
                                f"*Undo*. Use `/{bot_name} undo` if the conversation took a wrong turn or you cleared the history by accident.\n"
                                "\n"
                                "Please be patient with me, I am in an early stage of development."
                            )
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")

if __name__ == '__main__':
    app.start(port)
