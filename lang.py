import re

# tokens
TOKEN_PATTERNS = [
    ('NUMBER',  r'\d+(\.\d+)?'),
    ('IDENT',   r'[a-zA-Z_]\w*'),
    ('DCARET',  r'\^\^'),
    ('CARET',   r'\^'),
    ('PLUS',    r'\+'),
    ('MINUS',   r'-'),
    ('STAR',    r'\*'),
    ('SLASH',   r'/'),
    ('EQEQ',    r'=='),
    ('NEQ',     r'!='),
    ('GTE',     r'>='),
    ('LTE',     r'<='),
    ('GT',      r'>'),
    ('LT',      r'<'),
    ('EQUALS',  r'='),
    ('LPAREN',  r'\('),
    ('RPAREN',  r'\)'),
    ('COLON',   r':'),
    ('COMMA',   r','),
    ('SKIP',    r'[ \t]+'),
]

KEYWORDS = {'if', 'else', 'while', 'def', 'return'}

def lex(source):
    tokens = []
    i = 0
    while i < len(source):
        for name, pattern in TOKEN_PATTERNS:
            m = re.match(pattern, source[i:])
            if m:
                if name != 'SKIP':
                    text = m.group()
                    # promote keywords from IDENT
                    if name == 'IDENT' and text in KEYWORDS:
                        tokens.append((text.upper(), text))
                    else:
                        tokens.append((name, text))
                i += m.end()
                break
        else:
            raise SyntaxError(f"Unexpected character: '{source[i]}'")
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type=None):
        tok = self.tokens[self.pos]
        if expected_type and tok[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got '{tok[1]}'")
        self.pos += 1
        return tok

    def match(self, *types):
        tok = self.peek()
        return tok and tok[0] in types

    # ── statements ──

    def parse_statement(self):
        if self.match('IF'):
            return self.parse_if()
        if self.match('WHILE'):
            return self.parse_while()
        if self.match('DEF'):
            return self.parse_def()
        if self.match('RETURN'):
            return self.parse_return()
        return self.parse_expression()

    def parse_if(self):
        self.consume('IF')
        condition = self.parse_expression()
        self.consume('COLON')
        then_branch = self.parse_expression()
        else_branch = None
        if self.match('ELSE'):
            self.consume('ELSE')
            self.consume('COLON')
            else_branch = self.parse_expression()
        return ('if', condition, then_branch, else_branch)

    def parse_while(self):
        self.consume('WHILE')
        condition = self.parse_expression()
        self.consume('COLON')
        body = self.parse_expression()
        return ('while', condition, body)

    def parse_def(self):
        self.consume('DEF')
        name = self.consume('IDENT')[1]
        self.consume('LPAREN')
        params = []
        while not self.match('RPAREN'):
            params.append(self.consume('IDENT')[1])
            if self.match('COMMA'):
                self.consume('COMMA')
        self.consume('RPAREN')
        self.consume('COLON')
        body = self.parse_expression()
        return ('def', name, params, body)

    def parse_return(self):
        self.consume('RETURN')
        value = self.parse_expression()
        return ('return', value)

    # ── expressions ──

    def parse_expression(self):
        if self.match('IDENT'):
            next_tok = self.peek(1)
            if next_tok and next_tok[0] == 'EQUALS':
                name = self.consume('IDENT')[1]
                self.consume('EQUALS')
                value = self.parse_expression()
                return ('assign', name, value)
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_addition()
        if self.match('EQEQ', 'NEQ', 'GT', 'LT', 'GTE', 'LTE'):
            op = self.consume()[1]
            right = self.parse_addition()
            return ('binop', op, left, right)
        return left

    def parse_addition(self):
        left = self.parse_multiplication()
        while self.match('PLUS', 'MINUS'):
            op = self.consume()[1]
            right = self.parse_multiplication()
            left = ('binop', op, left, right)
        return left

    def parse_multiplication(self):
        left = self.parse_unary()
        while self.match('STAR', 'SLASH'):
            op = self.consume()[1]
            right = self.parse_unary()
            left = ('binop', op, left, right)
        return left

    def parse_unary(self):
        if self.match('MINUS'):
            self.consume()
            operand = self.parse_unary()
            return ('negate', operand)
        return self.parse_power()

    def parse_power(self):
        base = self.parse_call()
        if self.match('DCARET'):
            self.consume()
            exp = self.parse_power()
            return ('binop', '^^', base, exp)
        if self.match('CARET'):
            self.consume()
            exp = self.parse_power()
            return ('binop', '^', base, exp)
        return base

    def parse_call(self):
        expr = self.parse_primary()
        # if followed by LPAREN, it's a function call
        if self.match('LPAREN') and expr[0] == 'var':
            name = expr[1]
            self.consume('LPAREN')
            args = []
            while not self.match('RPAREN'):
                args.append(self.parse_expression())
                if self.match('COMMA'):
                    self.consume('COMMA')
            self.consume('RPAREN')
            return ('call', name, args)
        return expr

    def parse_primary(self):
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input")
        if tok[0] == 'NUMBER':
            self.consume()
            val = float(tok[1]) if '.' in tok[1] else int(tok[1])
            return ('num', val)
        if tok[0] == 'IDENT':
            self.consume()
            return ('var', tok[1])
        if tok[0] == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return ('group', expr)
        raise SyntaxError(f"Unexpected token: '{tok[1]}'")


def parse(tokens):
    parser = Parser(tokens)
    ast = parser.parse_statement()
    if parser.peek() is not None:
        raise SyntaxError(f"Unexpected token: '{parser.peek()[1]}'")
    return ast


# ── environment (supports nested scopes) ──

class Environment:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable '{name}'")

    def set(self, name, value):
        self.vars[name] = value

    def assign(self, name, value):
        # walk up the chain to find where it lives
        if name in self.vars:
            self.vars[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            self.vars[name] = value


# ── return value signal ──

class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


# ── evaluator ──

def evaluate(node, env):
    kind = node[0]

    if kind == 'num':
        return node[1]

    if kind == 'var':
        return env.get(node[1])

    if kind == 'assign':
        _, name, expr = node
        value = evaluate(expr, env)
        env.assign(name, value)
        return value

    if kind == 'binop':
        _, op, left, right = node
        l = evaluate(left, env)
        r = evaluate(right, env)
        if op == '+':  return l + r
        if op == '-':  return l - r
        if op == '*':  return l * r
        if op == '/':
            if r == 0: raise ZeroDivisionError("Division by zero")
            return l / r
        if op == '^':  return l ** r
        if op == '^^': return (l ** r) ** r
        if op == '==': return 1 if l == r else 0
        if op == '!=': return 1 if l != r else 0
        if op == '>':  return 1 if l > r else 0
        if op == '<':  return 1 if l < r else 0
        if op == '>=': return 1 if l >= r else 0
        if op == '<=': return 1 if l <= r else 0

    if kind == 'negate':
        return -evaluate(node[1], env)

    if kind == 'group':
        return evaluate(node[1], env)

    if kind == 'if':
        _, condition, then_branch, else_branch = node
        if evaluate(condition, env):
            return evaluate(then_branch, env)
        elif else_branch:
            return evaluate(else_branch, env)
        return None

    if kind == 'while':
        _, condition, body = node
        result = None
        while evaluate(condition, env):
            result = evaluate(body, env)
        return result

    if kind == 'def':
        _, name, params, body = node
        env.set(name, ('function', params, body, env))
        return f"<function {name}>"

    if kind == 'call':
        _, name, args = node
        func = env.get(name)
        if not isinstance(func, tuple) or func[0] != 'function':
            raise TypeError(f"'{name}' is not a function")
        _, params, body, closure_env = func
        if len(args) != len(params):
            raise TypeError(f"{name} expects {len(params)} args, got {len(args)}")
        # create a new scope for the function
        local_env = Environment(parent=closure_env)
        for param, arg in zip(params, args):
            local_env.set(param, evaluate(arg, env))
        try:
            return evaluate(body, local_env)
        except ReturnValue as r:
            return r.value

    if kind == 'return':
        raise ReturnValue(evaluate(node[1], env))

    raise RuntimeError(f"Unknown AST node: {kind}")


def run(source, env):
    tokens = lex(source)
    ast = parse(tokens)
    return evaluate(ast, env)


def repl():
    env = Environment()
    print("Mini Lang  |  type 'exit' to quit")
    print("-" * 35)
    while True:
        try:
            source = input("> ").strip()
            if not source: continue
            if source == 'exit': break
            result = run(source, env)
            if result is not None:
                print(result)
        except (SyntaxError, NameError, ZeroDivisionError, TypeError, RuntimeError) as e:
            print(f"Error: {e}")

repl()