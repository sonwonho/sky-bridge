import json

from env.config import GOOROOM_PROMPT_CONFIG
from utils.clova_completion_exe import CompletionExecutor
from utils.open_prompt import open_prompt


class Gooroom:
    def __init__(self):
        self.system_prompt = open_prompt("prompt/gooroom_system.txt")
        self.user_prompt = open_prompt("prompt/gooroom_user.txt")
        self.completion_executor = CompletionExecutor()
        self.request_data = dict(GOOROOM_PROMPT_CONFIG)

    def ask_schedule(self, year, subject, grade, university):
        preset_texts = []
        preset_texts.append({"role": "system", "content": self.system_prompt})
        preset_texts.append(
            {
                "role": "user",
                "content": self.user_prompt.format(
                    year=year, subject=subject, grade=grade, university=university
                ),
            }
        )
        request_data = self.request_data.copy()
        request_data["messages"] = preset_texts
        response_data = next(self.completion_executor.execute(request_data, "single"))
        response_data = json.loads(response_data)
        return response_data


if __name__ == "__main__":
    gr = Gooroom()
    print(gr.ask_schedule(2027, "수학", 2, "고려대"))
