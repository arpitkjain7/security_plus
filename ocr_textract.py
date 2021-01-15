import boto3

client = boto3.client("textract", region_name="us-east-2")


def call_textract(image_path):
    with open(image_path, "rb") as file:
        image = file.read()
        image = bytearray(image)
    textract_response = client.analyze_document(
        Document={"Bytes": image}, FeatureTypes=["TABLES", "FORMS"]
    )
    blocks = textract_response.get("Blocks")
    return blocks


def filter(blocks):
    extracted_data = []
    for block in blocks:
        if block.get("BlockType") == "WORD":
            word = block.get("Text")
            confidence_score = block.get("Confidence")
            extracted_data.append((word, confidence_score))
    return extracted_data


def detect_text(image_path):
    # print(image_path)
    blocks = call_textract(image_path)
    # print(blocks)
    extracted_data = filter(blocks)
    # print(extracted_data)
    return extracted_data


# ocr_data = detect_text(
#     "/Users/arpitkjain/Desktop/Data/POC/number-plate/number-plate-detection/OCR/1610697476595/6.jpg"
# )
# print(ocr_data)
