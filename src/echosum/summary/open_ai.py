from openai import OpenAI
client = OpenAI()


def summarize_text(text:str) ->str :
    response = client.responses.create(
        model="gpt-4.1-mini-2025-04-14",
        instructions="You shall summarize the input text",
        input=text,
    )

    return response.output_text