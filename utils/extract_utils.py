import re


# 获取高压设备线缆规格
def extract_high_voltage_number(device):
    match = re.search(r'(\d+)(?=/)', device)
    return int(match.group(0)) if match else None


# 获取低压设备线缆规格
def extract_low_voltage_number(device):
    match = re.search(r'(\d+)$', device)
    return int(match.group(0)) if match else None


# 获取变压器规格
def extract_transformer_capacity(device):
    match = re.search(r'-(\d+)', device)
    return float(match.group(1)) if match else None


# 获取电容器规格
def extract_capacitor_capacity(device):
    match = re.search(r'^(\d+).*(kvar|kVar)$', device)
    return int(match.group(1)) if match else None


# 获取分补容量
def extract_separated_total_capacity(device):
    if '分补' in device or '共补' in device:
        match = re.search(r'(\d+).(\d+)', device)
        return int(match.group(1)) * int(match.group(2)) if match else None

    # 正则匹配模板
    first_number_pattern = r'^\d+'
    number_before_bsmj_pattern = r'^(\d+).*(\d+)'
    second_last_number_pattern = r'-(\d+)-(\d+).*$'

    # 第一个数字
    first_number = re.search(first_number_pattern, device)
    first_number_value = int(first_number.group(0)) if first_number else None
    print(f'first_number: {first_number_value}')

    # 末尾倒数第二个数字
    second_last_number = re.search(second_last_number_pattern, device)
    second_last_number_value = int(second_last_number.group(1)) if second_last_number else None
    print(f'second_last_number: {second_last_number_value}')

    # 判断"BSMJ"前是否还有数字并匹配
    bsmj = 'BSMJ'
    if 'BSMJ' not in device:
        bsmj = 'BCMJ'
    bsmj_index = device.find(bsmj)
    before_bsmj = device[:bsmj_index]
    number_before_bsmj = re.search(number_before_bsmj_pattern, before_bsmj)
    number_before_bsmj_value = int(number_before_bsmj.group(2)) if number_before_bsmj else 1
    print(f'number_before_bsmj: {number_before_bsmj_value}')

    return first_number_value * second_last_number_value * number_before_bsmj_value
