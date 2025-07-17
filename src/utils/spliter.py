import csv
import pandas as pd
import re

# Load CSV
data = pd.read_csv("data/raw/Book1.csv", encoding="utf-8")

# Track problematic rows
problem_rows = []

# Split English description
def split_english_desc(desc):
    if pd.isna(desc) or str(desc).strip() == "":
        return "", "", ""
    sections = re.split(r'(Leadership motivation|Social Intelligence|Core Strengths|Key Strengths)', str(desc))
    result = {"Leadership motivation": "", "Social Intelligence": "", "Key Strengths": ""}
    try:
        for i in range(1, len(sections), 2):
            key = sections[i].replace(":", "").strip()
            val = sections[i+1].strip()
            if key == "Core Strengths":  # unify naming
                key = "Key Strengths"
            result[key] = val
    except Exception as e:
        print(f"Failed to parse English description: {desc}\nError: {e}")
    return result["Leadership motivation"], result["Social Intelligence"], result["Key Strengths"]

# Split Arabic description
def split_arabic_desc(desc):
    if pd.isna(desc) or str(desc).strip() == "":
        return "", "", ""
    desc = str(desc)
    desc = desc.replace("：", ":")
    desc = desc.replace(" :", ":")
    desc = re.sub(r"\s*:\s*", ":", desc)

    pattern = r'(التحفيز القيادي:|الذكاء العاطفي والاجتماعي|نقاط القوة الأساسية:|نقاط القوة:)'
    result = {"التحفيز القيادي": "", "الذكاء العاطفي والاجتماعي": "", "نقاط القوة الأساسية": ""}
    try:
        splits = re.split(pattern, desc)
        for i in range(1, len(splits), 2):
            key = splits[i].replace(":", "").strip()
            val = splits[i+1].strip()
            if key == "نقاط القوة":
                key = "نقاط القوة الأساسية"
            result[key] = val
    except Exception as e:
        print(f"Failed to split Arabic description: {desc}\nError: {e}")
    return result["التحفيز القيادي"], result["الذكاء العاطفي والاجتماعي"], result["نقاط القوة الأساسية"]

# Process and collect output
output_rows = []

for idx, row in data.iterrows():
    letters = row.get("letters", "").strip()

    eng_desc = row.get("english Description", "")
    ar_desc = row.get("arabic Description", "")

    if pd.isna(eng_desc) or pd.isna(ar_desc) or eng_desc.strip() == "" or ar_desc.strip() == "":
        problem_rows.append({"row": idx + 2, "letters": letters, "issue": "Empty description"})
        eng_lead, eng_soc, eng_key = "", "", ""
        ar_lead, ar_soc, ar_key = "", "", ""
    else:
        eng_lead, eng_soc, eng_key = split_english_desc(eng_desc)
        ar_lead, ar_soc, ar_key = split_arabic_desc(ar_desc)

    output_rows.append({
        "letters": letters,
        "Leadership_Motivation_en": eng_lead,
        "Emotional_Social_Intelligence_en": eng_soc,
        "Key_Strengths_Applications_en": eng_key,
        "Leadership_Motivation_ar": ar_lead,
        "Emotional_Social_Intelligence_ar": ar_soc,
        "Key_Strengths_Applications_ar": ar_key,
    })

# Write clean CSV
output_path = "data/processed/converted_output.csv"
with open(output_path, "w", newline='', encoding="utf-8") as f:
    fieldnames = [
        "letters",
        "Leadership_Motivation_en", "Emotional_Social_Intelligence_en", "Key_Strengths_Applications_en",
        "Leadership_Motivation_ar", "Emotional_Social_Intelligence_ar", "Key_Strengths_Applications_ar"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

# Optional: Log issues
if problem_rows:
    print(f"\n Found {len(problem_rows)} rows with empty descriptions:")
    for issue in problem_rows:
        print(f" - Row {issue['row']} (letters: {issue['letters']}): {issue['issue']}")

print(f"\nCSV file has been written to: '{output_path}'")
