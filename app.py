import os
from flask import Flask

template_dir = os.path.abspath("templates")
app = Flask(__name__, template_folder=template_dir, static_folder="static")

import chat

app.add_url_rule('/', view_func=chat.index)
app.add_url_rule('/api/chat', methods=['POST'], view_func=chat.chat)

if __name__ == "__main__":
    app.run(debug=True)


