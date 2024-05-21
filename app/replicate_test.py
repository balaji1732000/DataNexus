import replicate
import os

os.environ["REPLICATE_API_TOKEN"] = "r8_f4MP4W51W2lwXi8MLRZMlC0MshE3XU03gnIMi"
input = {
    "prompt": "Who are you?",
}
output = replicate.run("snowflake/snowflake-arctic-instruct", input=input)

print("".join(output))
