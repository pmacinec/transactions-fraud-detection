def split_pascal_case_string(input_str):
    """
    Split the string which is formatted in "PascalCase" to "Pascal Case"
    :param input_str: a string to be splitted
    :return: new string
    """
    output_str = ""
    
    for index, char in enumerate(input_str):
        low_char = char.lower()

        if low_char == char or not index:
            output_str += char
        else:
            output_str += f" {char}"
            
    return output_str