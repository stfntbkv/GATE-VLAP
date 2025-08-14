import re


def convert_words_to_digits(input_string):
    # Define a dictionary to map words to digits
    word_to_digit = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9'
    }

    # Use regular expression to find words that represent digits and replace them
    # with their corresponding numerical digits
    result_string = re.sub(r'\b(?:' + '|'.join(word_to_digit.keys()) + r')\b', lambda match: word_to_digit[match.group(0)], input_string)

    return result_string

def digits_to_letters(word):
    # Mapping of digits to their corresponding word form
    digit_to_word_map = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }

    # Check if the first character is a digit
    if word and word[0].isdigit():
        return digit_to_word_map[word[0]] + word[1:]
    return word

if __name__ == "__main__":
    print(digits_to_letters("7up0"))