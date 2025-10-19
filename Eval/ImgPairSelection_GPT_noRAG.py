import json
import os


data_file = "./TestSet_VQA_ReportGeneration_Final.json"

with open(data_file, 'r', encoding='utf-8') as f:
    historical_data = json.load(f)

print(len(historical_data))



Task3_Data_file = "./TestSet_ImagePairSelection.json"

with open(Task3_Data_file, 'r', encoding='utf-8') as f:
    Task3_Data = json.load(f)

print(len(Task3_Data))




############################################################################



from tqdm import tqdm
import base64
from openai import OpenAI



# OpenAI API Key
client = OpenAI(api_key="Your_api_key")




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


Answer_two_IMG = []

acc_num = 0
ALL_num = 0

yes_num = 0
unsure_num = 0



answers_file = "./ImgPairSelection_noRAG.jsonl"
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")



for i in tqdm(range(len(Task3_Data))):


    label_choice = Task3_Data[i]["right_option"]


    condition_change = Task3_Data[i]["condition_change"]

    option_A_idx = Task3_Data[i]["option"]["A"]
    option_B_idx = Task3_Data[i]["option"]["B"]
    option_C_idx = Task3_Data[i]["option"]["C"]

    print(Task3_Data[i]["option"])




    p1_A = historical_data[option_A_idx]["image_path"].replace('jpg', 'png')
    A_Now_img_path = f"./CheXpertPlus/PNG/{p1_A}"
    A_Now_img_base64 = encode_image(A_Now_img_path)

    p2_A = historical_data[option_A_idx]["historical_image_path"].replace('jpg', 'png')
    A_Hist_img_path = f"./CheXpertPlus/PNG/{p2_A}"
    A_Hist_img_base64 = encode_image(A_Hist_img_path)




    p1_B = historical_data[option_B_idx]["image_path"].replace('jpg', 'png')
    B_Now_img_path = f"./CheXpertPlus/PNG/{p1_B}"
    B_Now_img_base64 = encode_image(B_Now_img_path)

    p2_B = historical_data[option_B_idx]["historical_image_path"].replace('jpg', 'png')
    B_Hist_img_path = f"./CheXpertPlus/PNG/{p2_B}"
    B_Hist_img_base64 = encode_image(B_Hist_img_path)




    p1_C = historical_data[option_C_idx]["image_path"].replace('jpg', 'png')
    C_Now_img_path = f"./CheXpertPlus/PNG/{p1_C}"
    C_Now_img_base64 = encode_image(C_Now_img_path)

    p2_C = historical_data[option_C_idx]["historical_image_path"].replace('jpg', 'png')
    C_Hist_img_path = f"./CheXpertPlus/PNG/{p2_C}"
    C_Hist_img_base64 = encode_image(C_Hist_img_path)




    # --------------------------------------------------------------------------------
    


    # Prompt
    Text1 = f"\nYou are a professional radiologist. You are provided with three pairs of X-ray images. Each pair contains two X-ray images from the same patient. The first image in each pair is from the last visit, and the second one is from the current visit. Your task is to choose one of the options, based on the condition change from the last to the current visit, that best matches the following medical statement: '{condition_change}'. Please provide your answer by selecting the corresponding letter from the given choices. Please provide your final answer in the format: 'My answer is [option]' at the end of your response."


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A: ",},
                {"type": "text", "text": "\nLast visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{A_Hist_img_base64}"},},
                {"type": "text", "text": "\nCurrent visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{A_Now_img_base64}"},},

                {"type": "text", "text": "\nB: ",},
                {"type": "text", "text": "\nLast visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{B_Hist_img_base64}"},},
                {"type": "text", "text": "\nCurrent visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{B_Now_img_base64}"},},

                {"type": "text", "text": "\nC: ",},
                {"type": "text", "text": "\nLast visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{C_Hist_img_base64}"},},
                {"type": "text", "text": "\nCurrent visit image: ",},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{C_Now_img_base64}"},},
                
                {"type": "text", "text": Text1,},

            ],
        }
    ]



    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
    )


    print(response.choices[0].message.content)


    ans_file.write(json.dumps({
                            "data_idx": i,
                            "label": label_choice,
                            "answer": response.choices[0].message.content,
                            "condition_change": condition_change,
                            "prompt": Text1,
                            }) + "\n")
    ans_file.flush()


ans_file.close()






