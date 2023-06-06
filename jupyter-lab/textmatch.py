import Levenshtein

# 相似匹配
def match_similar_address(input_address, addresses, threshold):
    matched_addresses = []
    for address in addresses:
        similarity = Levenshtein.ratio(input_address, address)
        if similarity >= threshold:
            matched_addresses.append(address)
    return matched_addresses

# 示例用法  已有的Excel的地址拉出来转换下面格式
addresses = ["123 Main Street", "456 Elm Street", "789 Oak Street"]

input_address = "789 Street"
threshold = 0.8

matched_addresses = match_similar_address(input_address, addresses, threshold)
print("Matched addresses:", matched_addresses)
