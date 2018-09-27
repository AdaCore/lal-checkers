import sys


class Logger(object):
    """
    Logger class used as an intermediate step before outputting messages.
    Allows controlling the output through filters.
    """
    def __init__(self, filters):
        """
        :param dict[str, file] filters: The keys describe the categories that
            are "allowed". Their corresponding value indicates where the
            messages belonging to that category will be printed.
        """
        self.filters = filters

    @staticmethod
    def with_std_output(filters):
        """
        Creates a Logger from a list of allowed categories, redirecting each
        of them to the standard output.

        :param list[str] filters: The categories to allow.
        :rtype: Logger
        """
        return Logger({f: sys.stdout for f in filters})

    def log(self, category, msg):
        """
        Outputs the message according to its category.
        :param str category: Category of the message.
        :param str msg: Message content.
        """
        output = self.filters.get(category)
        if output is not None:
            output.write(msg)
            output.flush()

    def log_stdout(self, category):
        """
        Creates a logging context from this logger using the given category.

        :param str category: The category used to report messages inside the
            context.
        :rtype: _LoggingContext
        """
        return _LoggingContext(category, self)


class _LoggingContext(object):
    """
    Usage:

    with logging_context:
        print(MSG)

    Logging context are created by loggers through the "logging" method.
    """
    def __init__(self, category, logger):
        self.category = category
        self.logger = logger
        self.previous_stdout = None

    def __enter__(self):
        self.previous_stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.previous_stdout

    def write(self, text):
        self.logger.log(self.category, text)


# default logger outputs nothing
_global_logger = Logger({})


def set_logger(logger):
    """
    Sets the global logger object to the given Logger instance.
    :param Logger logger: The logger instance to use.
    """
    global _global_logger
    _global_logger = logger


def log(category, msg):
    """
    Logs the given message using the global logger instance, if defined.
    Note: adds a newline at the end of the message.

    :param str category: The category of the message.
    :param str msg: The content of the message.
    """
    return _global_logger.log(category, msg + '\n')


def log_stdout(category):
    """
    Creates a logging context from the global logger using the given category.
    :param str category: The category used to report messages inside the
        context.
    :rtype: _LoggingContext
    """
    return _global_logger.log_stdout(category)
