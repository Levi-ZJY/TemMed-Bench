import json
import os

with open("./TestSet_VQA_ReportGeneration_Final.json", 'r', encoding='utf-8') as f:
    historical_data = json.load(f)

print(len(historical_data))





from tqdm import tqdm
import base64
from openai import OpenAI




# OpenAI API Key
client = OpenAI(api_key="Your_api_key")




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")




answers_file = "./Report_GPT_Answer_RAG_imgReport.jsonl"
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")





for i in tqdm(range(len(historical_data))):


    img1 = historical_data[i]["image_path"].replace('jpg', 'png')
    image_path = f"./CheXpertPlus/PNG/{img1}"
    base64_image = encode_image(image_path)

    img2 = historical_data[i]["historical_image_path"].replace('jpg', 'png')
    his_image_path = f"./CheXpertPlus/PNG/{img2}"
    his_base64_image = encode_image(his_image_path)

    # --------------------------------------------------------------------------------


    REF_img1 = historical_data[i]["reference_images"][0]
    REF_image_path = REF_img1
    REF_base64_image = encode_image(REF_image_path)

    REF_img2 = historical_data[i]["hist_reference_images"][0]
    REF_his_image_path = REF_img2
    REF_his_base64_image = encode_image(REF_his_image_path)


    REF_report = historical_data[i]["reference_condition_changes_reports"][0]


    # --------------------------------------------------------------------------------


    Text1 = "You are a professional radiologist. You are provided with two reference X-ray images from the same patient, along with the corresponding report for the current visit image:"

    Text2 = f"\nReport for the current visit image: {REF_report}"

    Text3 = "\nPlease learn how to analyze X-ray images, track changes in the patient's condition, and generate reports based on this example.\n\nNow, you are given two new X-ray images from another patient:"

    Text4 = "\nThe first new image is from the last visit, and the second new image is from the current visit. Please generate a report for the new current visit image. You should consider the new last visit image to analyze the changes in the patient's condition in your report.\nNote that the diagnostic information from the reference images and report should not be directly used for diagnosis but only as a reference for comparison. You only need to generate the Impression section in the report. Please only include the content of the report in your response."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": Text1,},
                {"type": "text", "text": "\nLast visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{REF_his_base64_image}"},},
                {"type": "text", "text": "\nCurrent visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{REF_base64_image}"},},
                {"type": "text", "text": Text2,},
                {"type": "text", "text": Text3,},

                {"type": "text", "text": "\nLast visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{his_base64_image}"},},
                {"type": "text", "text": "\nCurrent visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},},
                {"type": "text", "text": Text4,},
            ],
        }
    ]



    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
    )


    Answer = {"id": i, "answer_report": response.choices[0].message.content, "label_report": historical_data[i]["condition_changes_report"]}
    

    ans_file.write(json.dumps(Answer) + "\n")

    ans_file.flush()


ans_file.close()






