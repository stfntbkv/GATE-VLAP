from chat_gpt_oam.logical_expressions.logical_expression import LogicExpression, AtomicExpression, OrExpression, \
    AndExpression, NotExpression

ANSWER_KEYWORD = "ANSWER:"


def extract_yes_no(answer: str) -> bool:
    index = answer.find(ANSWER_KEYWORD) + len(ANSWER_KEYWORD)
    final_answer = answer[index:]
    return "yes".casefold() in final_answer.casefold()


def parse_logic_expression(expression: str, model, history) -> LogicExpression:
    def parse_tokens(tokens):
        if len(tokens) == 1:
            return AtomicExpression(tokens[0], model=model,history=history)

        if '|' in tokens:
            index = tokens.index('|')
            left = parse_tokens(tokens[:index])
            right = parse_tokens(tokens[index + 1:])
            return OrExpression([left, right])

        if '&' in tokens:
            index = tokens.index('&')
            left = parse_tokens(tokens[:index])
            right = parse_tokens(tokens[index + 1:])
            return AndExpression([left, right])

        if '!' in tokens:
            index = tokens.index('!')
            right = parse_tokens(tokens[index + 1:])
            return NotExpression(right)

    def extract_subexpression(expression: str, start: int) -> str:
        open_parentheses = 1
        for idx, char in enumerate(expression[start + 1:]):
            if char == '(':
                open_parentheses += 1
            elif char == ')':
                open_parentheses -= 1

            if open_parentheses == 0:
                return expression[start + 1:start + 1 + idx]

    tokens = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char == '(':
            subexpression = extract_subexpression(expression, i)
            tokens.append(subexpression)
            i += len(subexpression) + 2
        elif char == '&' or char == '|' or char == '!':
            tokens.append(char)
            i += 1
        else:
            i += 1
    special_characters = ['(', ')', '&', '|', '!']
    if not any(char in expression for char in special_characters):
        tokens.append(expression)

    return parse_tokens(tokens)
