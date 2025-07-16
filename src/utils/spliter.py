import csv
import pandas as pd
import re

# Read the input CSV file
data = pd.read_csv("data/raw/result symbols with the new discriptions.csv", encoding="utf-8")

def split_english_desc(desc):
    sections = re.split(r'(Leadership motivation:|Social Intelligence:|Key Strengths:)', str(desc))
    result = {"Leadership motivation": "", "Social Intelligence": "", "Key Strengths": ""}
    for i in range(1, len(sections), 2):
        key = sections[i].replace(":", "").strip()
        val = sections[i+1].strip()
        result[key] = val
    return result["Leadership motivation"], result["Social Intelligence"], result["Key Strengths"]

def split_arabic_desc(desc):
    # Normalize colons and spaces
    desc = str(desc)
    desc = desc.replace("：", ":")  # Chinese fullwidth colon
    desc = desc.replace(" :", ":")  # Space before colon
    desc = re.sub(r"\s*:\s*", ":", desc)  # Remove spaces around colons

    # Define the split pattern here!
    pattern = r'(التحفيز القيادي:|الذكاء العاطفي والاجتماعي|نقاط القوة الأساسية:|نقاط القوة:)'
    try:
        splits = re.split(pattern, desc)
        result = {"التحفيز القيادي": "", "الذكاء العاطفي والاجتماعي": "", "نقاط القوة الأساسية": ""}
        for i in range(1, len(splits), 2):
            key = splits[i].replace(":", "").strip()
            val = splits[i+1].strip()
            # Fix: Handle alternative label
            if key == "نقاط القوة":
                key = "نقاط القوة الأساسية"
            result[key] = val
        return (result["التحفيز القيادي"], 
                result["الذكاء العاطفي والاجتماعي"],
                result["نقاط القوة الأساسية"])
    except Exception as e:
        print(f"Failed to split desc: {desc}\nError: {e}")
        return ("", "", "")

output_rows = []

for idx, row in data.iterrows():
    letters = row["letters"]
    eng_lead, eng_soc, eng_key = split_english_desc(row["english Description"])
    ar_lead, ar_soc, ar_key = split_arabic_desc(row["arabic Description"])
    output_rows.append({
        "letters": letters,
        "Leadership_Motivation_en": eng_lead,
        "Emotional_Social_Intelligence_en": eng_soc,
        "Key_Strengths_Applications_en": eng_key,
        "Leadership_Motivation_ar": ar_lead,
        "Emotional_Social_Intelligence_ar": ar_soc,
        "Key_Strengths_Applications_ar": ar_key,
    })

# Write to CSV
with open("data/processed/converted_output.csv", "w", newline='', encoding="utf-8") as f:
    fieldnames = [
        "letters",
        "Leadership_Motivation_en", "Emotional_Social_Intelligence_en", "Key_Strengths_Applications_en",
        "Leadership_Motivation_ar", "Emotional_Social_Intelligence_ar", "Key_Strengths_Applications_ar"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for outrow in output_rows:
        writer.writerow(outrow)

print("CSV file has been written as 'converted_output.csv'")
