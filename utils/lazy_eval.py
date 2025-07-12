class LazyEval:
    def __init__(self, expression, globals_=None, locals_=None):
        self.expression = expression
        self.globals = globals_ or globals()
        self.locals = locals_ or locals()
        self._value = None

    def _evaluate(self):
        self._value = eval(self.expression, self.globals, self.locals)
        self._evaluated = True

    def __repr__(self):
        self._evaluate()
        return repr(self._value)

    def __str__(self):
        self._evaluate()
        return str(self._value)

    def __getattr__(self, attr):
        self._evaluate()
        return getattr(self._value, attr)

    def __call__(self, *args, **kwargs):
        self._evaluate()
        return self._value(*args, **kwargs)
