from .open_ai import summarize_text as summ_open_ai

def summarize_text(text : str):
    return summ_open_ai(text=text)