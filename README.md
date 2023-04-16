# viccy
GPT integration for Slack

This is a work in progress...

## Slack app checklist

* Create app from scratch
* Basic Information --> App Credentials --> Signing Secret, save as environment variable `SLACK_SIGNING_SECRET`
* Basic Information --> Display Information --> Set name, description, icon, and color
* Event Subscriptions --> Enable Events --> On
* Start service with new `SLACK_SIGNING_SECRET`
* Event Subscriptions --> Enable Events --> Request URL (take URL from Heroku or ngrok and add `/slack/events`)
* Event Subscriptions --> Subscribe to bot events
* * `app_home_opened`
* * `app_mention`
* * `message.channels`
* * `message.groups`
* * `message.im`
* * `message.mpim`
* OAuth & Permissions --> Scopes
* * `channels:read`
* * `chat:write`
* * `chat:write.customize`
* * `files:read`
* * `groups:read`
* * `im:read`
* * `metadata.message:read`
* * `mpim:read`
* App Home --> Your App’s Presence in Slack --> Always Show My Bot as Online
* App Home --> Show Tabs --> Home Tab --> On
* App Home --> Show Tabs --> Messages Tab --> On
* App Home --> Show Tabs --> Messages Tab --> Allow users to send Slash commands and messages from the messages tab
* Slash Commands --> Create New Command
* * Command: `/viccy`
* * Request URL: Same as in Event Subscriptions
* * Short Description: `Control chat history`
* * Usage Hint: `[start, stop, status, reset or list]`
* Install App --> Install to Workspace
* Install App --> Bot User OAuth Token, save as environment variable `SLACK_BOT_TOKEN`
* Restart service with new `SLACK_BOT_TOKEN`
