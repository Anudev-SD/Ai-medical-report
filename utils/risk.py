def compute_risk(data):
    high = sum(1 for x in data if x["status"] == "High")
    low = sum(1 for x in data if x["status"] == "Low")

    score = high * 2 + low

    if score == 0:
        return "Low"
    elif score <= 2:
        return "Medium"
    else:
        return "High"