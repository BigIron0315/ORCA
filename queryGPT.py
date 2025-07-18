## queryGPT.py

import os
import subprocess
import json
from openai import OpenAI
import openai
import sys
from packaging import version

# === Version check ===
required_version = "1.0.0"
current_version = openai.__version__

if version.parse(current_version) < version.parse(required_version):
    sys.exit(f"❌ Incompatible OpenAI version: {current_version}. Please use >= {required_version}.\n"
             f"Try running: pip install --upgrade openai")


def Query_GPT4(system_msg, user_msg, temp):
    client = OpenAI(api_key="Your-API-Key")
    response = client.chat.completions.create(
        model="o4-mini",#"o4-mini",#"gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message.content