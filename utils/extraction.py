import re

def extract_medical_values(text):
    pattern = r'([A-Za-z ]+):\s*([\d.]+)\s*([a-zA-Z/]+)?\s*\((Low|High|Normal)?\)'
    
    matches = re.findall(pattern, text)
    
    results = []
    for m in matches:
        results.append({
            "test": m[0].strip(),
            "value": m[1],
            "unit": m[2],
            "status": m[3] if m[3] else "Unknown"
        })
    
    return results