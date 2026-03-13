def highlight_keywords(text, keywords):

    for word in keywords:

        text = text.replace(
            word,
            f"**{word}**"
        )

    return text