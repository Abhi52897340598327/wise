import re

# tokens
TOKEN_PATTERNS = [
    ('STRING',  r'"[^"]*"'),
    ('NUMBER',  r'\d+(\.\d+)?'),
    ('IDENT',   r'[a-zA-Z_]\w*'),
    ('DOT',     r'\.'),
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

KEYWORDS = {'if', 'else', 'while', 'def', 'return', 'print', 'class'}

def lex(source):
    tokens = []
    i = 0
    while i < len(source):
        for name, pattern in TOKEN_PATTERNS:
            m = re.match(pattern, source[i:])
            if m:
                if name != 'SKIP':
                    text = m.group()
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

    def parse_statement(self):
        if self.match('IF'):
            return self.parse_if()
        if self.match('WHILE'):
            return self.parse_while()
        if self.match('DEF'):
            return self.parse_def()
        if self.match('CLASS'):
            return self.parse_class()
        if self.match('RETURN'):
            return self.parse_return()
        if self.match('PRINT'):
            return self.parse_print()
        return self.parse_expression()

    def parse_if(self):
        self.consume('IF')
        condition = self.parse_expression()
        self.consume('COLON')
        then_branch = self.parse_statement()
        else_branch = None
        if self.match('ELSE'):
            self.consume('ELSE')
            self.consume('COLON')
            else_branch = self.parse_statement()
        return ('if', condition, then_branch, else_branch)

    def parse_while(self):
        self.consume('WHILE')
        condition = self.parse_expression()
        self.consume('COLON')
        body = self.parse_statement()
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
        body = self.parse_statement()
        return ('def', name, params, body)

    def parse_return(self):
        self.consume('RETURN')
        value = self.parse_expression()
        return ('return', value)
    
    def parse_print(self):
        self.consume('PRINT')
        self.consume('LPAREN')
        value = self.parse_expression()
        self.consume('RPAREN')
        return ('print', value)
    
    def parse_class(self):
        self.consume('CLASS')
        name = self.consume('IDENT')[1]
        self.consume('COLON')
        method = self.parse_def()
        return ('class', name, [method])

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
        
        # handle dot notation (method calls and attribute access)
        while self.match('DOT'):
            self.consume('DOT')
            method_name = self.consume('IDENT')[1]
            if self.match('LPAREN'):
                self.consume('LPAREN')
                args = []
                while not self.match('RPAREN'):
                    args.append(self.parse_expression())
                    if self.match('COMMA'):
                        self.consume('COMMA')
                self.consume('RPAREN')
                expr = ('method_call', expr, method_name, args)
            else:
                expr = ('get_attr', expr, method_name)
        
        # handle function calls
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
        if tok[0] == 'STRING':
            self.consume()
            text = tok[1][1:-1]
            return ('str', text)
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
        if name in self.vars:
            self.vars[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            self.vars[name] = value


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


def evaluate(node, env):
    kind = node[0]

    if kind == 'num':
        return node[1]
    
    if kind == 'str':
        return node[1]

    if kind == 'var':
        return env.get(node[1])

    if kind == 'assign':
        _, name, expr = node
        value = evaluate(expr, env)
        env.assign(name, value)
        return value
    
    if kind == 'print':
        _, value = node
        result = evaluate(value, env)
        print(result)
        return None

    if kind == 'binop':
        _, op, left, right = node
        l = evaluate(left, env)
        r = evaluate(right, env)
        if op == '+':
            if isinstance(l, str) or isinstance(r, str):
                return str(l) + str(r)
            return l + r
        if op == '-':  return l - r
        if op == '*':  return l * r
        if op == '/':
            if r == 0: raise ZeroDivisionError("Division by zero")
            return l / r
        if op == '^':  return l ** r
        if op == '^^': return (l ** r) ** r
        if op == '==': return 'true' if l == r else 'false'
        if op == '!=': return 'true' if l != r else 'false'
        if op == '>':  return 'true' if l > r else 'false'
        if op == '<':  return 'true' if l < r else 'false'
        if op == '>=': return 'true' if l >= r else 'false'
        if op == '<=': return 'true' if l <= r else 'false'


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
    
    if kind == 'class':
        _, name, methods = node
        env.set(name, ('class', methods, env))
        return f"<class {name}>"

    if kind == 'call':
        _, name, args = node
        func = env.get(name)
        
        # handle class instantiation
        if isinstance(func, tuple) and func[0] == 'class':
            _, methods, class_env = func
            obj = {'__class__': name, '__methods__': methods, '__env__': class_env}
            return ('object', obj)
        
        # handle function calls
        if not isinstance(func, tuple) or func[0] != 'function':
            raise TypeError(f"'{name}' is not a function")
        _, params, body, closure_env = func
        if len(args) != len(params):
            raise TypeError(f"{name} expects {len(params)} args, got {len(args)}")
        local_env = Environment(parent=closure_env)
        for param, arg in zip(params, args):
            local_env.set(param, evaluate(arg, env))
        try:
            return evaluate(body, local_env)
        except ReturnValue as r:
            return r.value
    
    if kind == 'object':
        return node[1]
    
    if kind == 'method_call':
        _, obj_expr, method_name, args = node
        obj = evaluate(obj_expr, env)
        if not isinstance(obj, tuple) or obj[0] != 'object':
            raise TypeError("Can only call methods on objects")

        _, obj_data = obj
        methods = obj_data['__methods__']

        method_def = None
        for m in methods:
            if m[0] == 'def' and m[1] == method_name:
                method_def = m
                break
        
        if not method_def:
            raise AttributeError(f"Object has no method '{method_name}'")
        
        _, _, params, body = method_def
        if len(args) != len(params):
            raise TypeError(f"{method_name} expects {len(params)} args, got {len(args)}")
        
        method_env = Environment(parent=obj_data['__env__'])
        method_env.set('self', ('object', obj_data))
        for param, arg in zip(params, args):
            method_env.set(param, evaluate(arg, env))

        try:
            return evaluate(body, method_env)
        except ReturnValue as r:
            return r.value
    
    if kind == 'get_attr':
        _, obj_expr, attr_name = node
        obj = evaluate(obj_expr, env)
        if not isinstance(obj, tuple) or obj[0] != 'object':
            raise TypeError("Can only access attributes on objects")
        _, obj_data = obj
        if attr_name in obj_data:
            return obj_data[attr_name]
        raise AttributeError(f"Object has no attribute '{attr_name}'")
    
    if kind == 'return':
        raise ReturnValue(evaluate(node[1], env))

    raise RuntimeError(f"Unknown AST node: {kind}")


def run(source, env):
    tokens = lex(source)
    ast = parse(tokens)
    return evaluate(ast, env)


def repl():
    env = Environment()
    print("Wise  |  type 'exit' to quit")
    print("-" * 35)
    while True:
        try:
            source = input("> ").strip()
            if not source: continue
            if source == 'exit': break
            result = run(source, env)
            if result is not None:
                print(result)
        except (SyntaxError, NameError, ZeroDivisionError, TypeError, RuntimeError, AttributeError) as e:
            print(f"Error: {e}")

repl()