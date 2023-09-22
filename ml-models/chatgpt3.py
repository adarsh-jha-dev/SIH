import openai
import gradio

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a psychiatrist"}]

def CustomChatGPT(type_your_message_here):
    messages.append({"role": "user", "content": type_your_message_here})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Chatbot")

demo.launch(share=True)
